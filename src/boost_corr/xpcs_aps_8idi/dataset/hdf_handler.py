import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# required for bloc compression
import hdf5plugin
import h5py
import torch
import logging
from .xpcs_dataset import XpcsDataset
from .hdf_direct import (BatchPrefetcher, FrameDecoder, acquire_staging,
                         cast_like_hdf5, probe_direct_source, release_staging)


logger = logging.getLogger(__name__)

# decompression threads for direct read. More than this only helps when the
# host-to-device copy is not the bottleneck.
DEFAULT_NUM_THREADS = 32
# direct read crops in the decompression threads when mask_crop keeps at most
# this fraction of the pixels, and on the device otherwise: gathering on the
# host costs more than copying the full frame once the crop gets large.
HOST_CROP_RATIO = 0.25


class HdfDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
    direct_read: bool
        read frame-per-chunk bitshuffle/uncompressed data with a pool of
        threads that bypasses h5py (see hdf_direct); batches are then
        returned on the device. Files that don't qualify, and all files when
        the environment variable BOOST_CORR_HDF_DIRECT is 0, are read with
        h5py and are meant to be wrapped in a DataLoader.
    num_threads: int
        number of decompression threads for direct read. None for auto.
    """
    def __init__(self,
                 *args,
                 preload_size=8,
                 dtype=np.uint8,
                 data_path='/entry/data/data',
                 direct_read=True,
                 num_threads=None,
                 **kwargs):

        super(HdfDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "HDF5 Dataset"
        self.is_sparse = False
        with h5py.File(self.fname, 'r') as f:
            data = f[data_path]
            self.shape = data.shape

            # update data type;
            if data.dtype == np.uint8:
                self.dtype = np.int16
            elif data.dtype == np.uint16:
                # lambda2m's uint16 is actually 12bit. it's safe to to int16
                if self.shape[1] * self.shape[2] == 1813 * 1558:
                    self.dtype = np.int16
                # likely eiger detectors
                else:
                    self.dtype = np.int32
            elif data.dtype == np.uint32:
                self.dtype = np.int32
                logger.warn('cast uint32 to int32. it may cause ' + 
                            'overflow when the maximal value >= 2^31')
            else:
                self.dtype = data.dtype
        
            if kwargs['avg_frame'] > 1:
                self.dtype = np.float32

            self.source = None
            if direct_read and os.environ.get('BOOST_CORR_HDF_DIRECT', '1') != '0':
                self.source = self.probe_direct(data)

        self.direct_read = self.source is not None
        self.torch_dtype = None
        if self.direct_read:
            self.torch_dtype = torch.from_numpy(np.empty(0, dtype=self.dtype)).dtype
        if num_threads is None:
            if hasattr(os, 'sched_getaffinity'):
                cpu_count = len(os.sched_getaffinity(0))
            else:
                cpu_count = os.cpu_count() or 1
            num_threads = min(DEFAULT_NUM_THREADS, cpu_count)
        self.num_threads = num_threads
        if self.direct_read:
            logger.info(f'direct read ({self.source.describe()}) with '
                        f'{self.num_threads} threads')
        self.decoder = None
        self.pool = None
        self.staging = None
        self.prefetcher = None
        self.next_batch = 0
        self.device_crop = None

        self.fhdl = None
        self.data = None
        self.data_cache = None
        self.data_path = data_path

        self.update_batch_info(self.shape[0])
        self.update_det_size(self.shape[1:])

        if self.mask_crop is not None:
            self.mask_crop = self.mask_crop.cpu().numpy()

        self.current_group = None
        self.preload_size = preload_size

    def probe_direct(self, data):
        """
        return the DirectSource of the dataset if it qualifies for direct
        read and the first frame decodes to what h5py returns; else None
        """
        try:
            source = probe_direct_source(data)
            if source is None:
                return None
            frame = np.empty(self.shape[1:], dtype=data.dtype)
            decoder = FrameDecoder(source)
            try:
                decoder.decode(0, frame.reshape(-1))
            finally:
                decoder.close()
            if not np.array_equal(frame, data[0]):
                raise ValueError('the first frame differs from the h5py read')
        except Exception as e:
            logger.warning(f'direct read is not usable, falling back to h5py: {e}')
            return None
        return source

    def __reset__(self):
        if self.prefetcher is not None:
            self.prefetcher.stop()
            self.prefetcher = None
        if self.fhdl is not None:
            self.fhdl.close()
            self.data = None
            self.fhdl = None

    def close(self):
        self.__reset__()
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None
        if self.decoder is not None:
            self.decoder.close()
            self.decoder = None
        self.release_staging()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_frames(self, idx):
        beg, end, _ = self.get_raw_index(idx)
        return range(beg, end, self.stride)

    def start_prefetch(self, idx):
        self.__reset__()
        if self.pool is None:
            self.decoder = FrameDecoder(self.source)
            self.pool = ThreadPoolExecutor(self.num_threads, thread_name_prefix='hdf-decode')

        crop = self.mask_crop
        width = self.pixel_num
        self.device_crop = None
        if crop is not None:
            if crop.min() < 0 or crop.max() >= self.pixel_num:
                raise IndexError('mask_crop is out of the range of the detector pixels')
            if crop.shape[0] <= HOST_CROP_RATIO * self.pixel_num:
                width = crop.shape[0]
            else:
                self.device_crop = torch.as_tensor(crop, device=self.device)
                crop = None

        # raw frames per batch
        size = self.batch_size * self.avg_frame
        # enough batches in flight to keep every thread busy
        count = math.ceil(self.num_threads / size) + 2
        key = (count, (size, width), self.source.staging_dtype,
               torch.device(self.device).type == 'cuda')
        if self.staging is None or self.staging[0] != key:
            self.release_staging()
            self.staging = (key, acquire_staging(key))

        batches = [self.get_frames(n) for n in range(idx, self.batch_num)]
        self.prefetcher = BatchPrefetcher(self.decoder, batches, self.staging[1],
                                          self.pool, self.device, crop=crop)

    def release_staging(self):
        if self.staging is not None:
            release_staging(*self.staging)
            self.staging = None

    def getbatch_direct(self, idx):
        """
        return tensor on the device; batches are prefetched in order, so
        sequential access is fast and any other access restarts the prefetch
        """
        if not 0 <= idx < self.batch_num:
            raise IndexError(f'batch index {idx} is out of range')
        if self.prefetcher is None or idx != self.next_batch:
            self.start_prefetch(idx)
        try:
            x = self.prefetcher.get()
        except BaseException:
            # the prefetch has ended; start over on the next access
            self.__reset__()
            raise
        self.next_batch = idx + 1
        x = cast_like_hdf5(x, self.source.kind, self.torch_dtype)
        if self.device_crop is not None:
            x = x[:, self.device_crop]
        return x

    def __getbatch__(self, idx):
        """
        return numpy array, which will be converted to tensor with dataloader
        """
        if self.direct_read:
            return self.getbatch_direct(idx)

        if self.fhdl is None:
            self.fhdl = h5py.File(self.fname, 'r', rdcc_nbytes=1024*1024*256)
            self.data = self.fhdl[self.data_path]

        beg, end, size = self.get_raw_index(idx)
        # a new buffer for every batch. A DataLoader worker pickles the batch
        # it returned in a background thread while the next one is already
        # being read, so a buffer shared between batches hands out the
        # frames of the wrong batch.
        x = np.empty(shape=(size, self.shape[1], self.shape[2]), dtype=self.dtype)
        # idx_list = np.arange(beg, end, self.stride)

        self.data.read_direct(x, np.s_[beg:end:self.stride])
        x = x.reshape(size, self.pixel_num)
        if self.mask_crop is not None:
            x = x[:, self.mask_crop]

        return torch.from_numpy(x)


def test():
    fname = (
        "/clhome/MQICHU/ssd/xpcs_data_raw/A003_Cu3Au_att0_001/A003_Cu3Au_att0_001.imm"
    )
    ds = HdfDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape, ds[n].device)


def test_bin(idx):
    fname = f"/clhome/MQICHU/ssd/APSU_TestData_202106/APSU_TestData_{idx:03d}/APSU_TestData_{idx:03d}.h5"
    ds = HdfDataset(fname)
    ds.to_rigaku_bin(f"hdf2bin_{idx:03d}.bin")


if __name__ == '__main__':
    test_bin(4)
