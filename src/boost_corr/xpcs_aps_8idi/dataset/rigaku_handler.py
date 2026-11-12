import numpy as np
import logging
import torch
from .xpcs_dataset import XpcsDataset
from .rigaku_utils import convert_sparse
from ...devices import is_gpu_device
import os


logger = logging.getLogger(__name__)


class RigakuDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    """

    def __init__(self, *args, dtype=np.uint8, total_frames=None, **kwargs):
        super(RigakuDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.ifc = self.read_data(total_frames)
        self.ifc = self.to_device()
        self._mem_addr = None

    def to_device(self, max_fsize_gb_in_gpu=1.0):
        fsize = os.path.getsize(self.fname) / 1024**3
        if fsize <= max_fsize_gb_in_gpu and is_gpu_device(self.device):
            device = self.device
        else:
            device = "cpu"
            logger.info(
                f"{os.path.basename(self.fname)}: {fsize:.2f} GB, kept in CPU RAM"
            )

        index = torch.tensor(self.ifc[0].astype(np.int32), device=device)
        frame = torch.tensor(self.ifc[1].astype(np.int32), device=device)
        count = torch.tensor(self.ifc[2].astype(np.uint8), device=device)
        return (index, frame, count)

    def read_data(self, total_frames=None):
        with open(self.fname, "r") as f:
            a = np.fromfile(f, dtype=np.uint64)
            d = convert_sparse(a)
            index, frame, count = d[0], d[1], d[2]
            if total_frames is None:
                total_frames = frame[-1] + 1
            else:
                total_frames = max(total_frames, frame[-1] + 1)

        # fix Rigaku3m module index offset
        if np.min(index) >= 1024 * 1024:
            index -= 1024 * 1024

        # update frame_num and batch_num
        self.update_batch_info(total_frames)

        # mem_addr (the per-frame begin/end slice table) is built lazily --
        # see the mem_addr property below. It is only needed by
        # __getbatch__, the dense-frame path multitau/twotime use; the
        # event correlator's get_events() never touches it. For finely
        # time-binned data (frame_num_raw in the hundreds of millions to
        # billions -- e.g. a nanosecond-binned RigakuExtend/.bix file) the
        # np.arange + searchsorted + list-of-slices below can take tens of
        # seconds to minutes, for no benefit if __getbatch__ is never called.
        self._frame_for_mem_addr = frame
        return (index, frame, count)

    @property
    def mem_addr(self):
        """Per-frame (begin, end) slice into the frame-sorted event arrays,
        built lazily on first access -- see the note in read_data()."""
        if self._mem_addr is None:
            frame = self._frame_for_mem_addr
            frame_num_raw = self.frame_num_raw
            all_index = np.arange(0, frame_num_raw + 2)
            all_index[-1] = frame_num_raw
            beg = np.searchsorted(frame, all_index[:-1])
            end = np.searchsorted(frame, all_index[1:])
            self._mem_addr = [slice(a, b) for a, b in zip(beg, end)]
        return self._mem_addr

    def __getbatch__(self, idx):
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        device = self.ifc[0].device
        x = torch.zeros(size, self.pixel_num, dtype=torch.uint8, device=device)
        if self.stride == 1:
            # the data is continuous in RAM; convert by batch
            sla, slb = self.mem_addr[beg], self.mem_addr[end]
            a, b = sla.start, slb.start
            x[self.ifc[1][a:b].long() - beg, self.ifc[0][a:b].long()] = self.ifc[2][a:b]
        else:
            # the data is discrete in RAM; convert frame by frame
            for n, idx in enumerate(np.arange(beg, end, self.stride)):
                sl = self.mem_addr[idx]
                x[n, self.ifc[0][sl].long()] = self.ifc[2][sl]
        x = x.to(self.device)
        # disabled for now: signal-spill correction is superseded by the
        # event correlator's CCF direction mechanism when --signal-spill is
        # set (see EventCorrelator). Kept, not deleted, for reference/future
        # reactivation.
        # x = self.update_signal_spill(x.reshape(size, *self.det_size))
        x = x.reshape(size, self.pixel_num)
        if self.mask_crop is not None:
            x = x[:, self.mask_crop]
        return x

    def get_events(self):
        """Return (pixel, time_bin, count, bin_width_ns) for the event
        correlator: int64 numpy arrays over the full (uncropped) pixel
        space, sorted by time_bin (as stored in self.ifc). Inherited
        unchanged by RigakuExtendDataset -- it overrides read_data() only,
        so self.ifc is already built from the correct bit layout by the
        time this runs.
        """
        # imported lazily: dataset/utils.py imports RigakuDataset at module
        # level, so a top-level import here would be circular.
        from .utils import resolve_bin_width_ns

        pixel = self.ifc[0].detach().cpu().numpy().astype(np.int64)
        time_bin = self.ifc[1].detach().cpu().numpy().astype(np.int64)
        count = self.ifc[2].detach().cpu().numpy().astype(np.int64)
        bin_width_ns = resolve_bin_width_ns(self.fname, self.bin_time_s)
        return pixel, time_bin, count, bin_width_ns


def test():
    fname = (
        "/home/beams/MQICHU/ramdisk/O018_Silica_D100_att0_Rq0_00001/"
        "O018_Silica_D100_att0_Rq0_00001.bin"
    )
    ds = RigakuDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape)


if __name__ == "__main__":
    test()
