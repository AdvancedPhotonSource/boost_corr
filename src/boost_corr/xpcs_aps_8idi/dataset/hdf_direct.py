"""Direct-chunk reader for frame-per-chunk HDF5 datasets.

h5py serializes every read behind a global lock, so decompressing an
Eiger-style bitshuffle file frame by frame is single threaded per process,
and fanning out to DataLoader worker processes pays for it again by shipping
each decompressed batch back through shared memory. This module bypasses the
HDF5 library for the pixel data: the chunk index is read once, then worker
threads `pread` the compressed chunks and decompress them with the bitshuffle
library bundled in hdf5plugin (called through ctypes, which releases the GIL)
straight into page-locked staging buffers that are copied to the device.

Only layouts that can be decoded without the HDF5 filter pipeline qualify
(see `probe_direct_source`); anything else stays on the h5py path.
"""

import ctypes
import glob
import logging
import os
import queue
import struct
import threading
from collections import deque

import hdf5plugin
import numpy as np
import torch

logger = logging.getLogger(__name__)


BSHUF_FILTER_ID = 32008
# cd_values[4] of the bitshuffle HDF5 filter -> decompress function exported
# by the filter library
_BSHUF_DECODERS = {2: "bshuf_decompress_lz4", 3: "bshuf_decompress_zstd"}
# chunk header written by the bitshuffle HDF5 filter (big endian):
# uncompressed size in bytes (uint64), block size in bytes (uint32)
_BSHUF_HEADER = struct.Struct(">QI")

# file dtype -> torch dtype of the staging buffer. torch has no general
# uint16/uint32 support, so those are staged as the signed type of the same
# width (the decoder only writes bytes) and reinterpreted in cast_like_hdf5.
_STAGING_DTYPE = {
    "u1": torch.uint8,
    "u2": torch.int16,
    "u4": torch.int32,
    "i2": torch.int16,
    "i4": torch.int32,
    "f4": torch.float32,
}

_bshuf_lib = None


def _load_bshuf():
    """Return the bitshuffle filter library bundled in hdf5plugin, or None."""
    global _bshuf_lib
    if _bshuf_lib is None:
        try:
            path = glob.glob(os.path.join(hdf5plugin.PLUGIN_PATH, "libh5bshuf.*"))[0]
            lib = ctypes.CDLL(path)
            for name in _BSHUF_DECODERS.values():
                func = getattr(lib, name)
                # int64_t f(const void* in, void* out, size_t size,
                #           size_t elem_size, size_t block_size)
                func.restype = ctypes.c_int64
                func.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * 3
            _bshuf_lib = lib
        except (AttributeError, IndexError, OSError) as e:
            logger.info(f"bitshuffle library is not usable for direct read: {e}")
            _bshuf_lib = False
    return _bshuf_lib or None


class DirectSource(object):
    """Where the frames of one dataset live in its file, and how to decode
    them. `addr`/`nbytes` are indexed by frame; `decoder` is the name of the
    bitshuffle decompress function, or None when the frames are stored raw;
    `unfiltered` flags individual chunks the writer stored without the filter.
    """

    def __init__(self, fname, kind, frame_size, addr, nbytes, decoder=None, unfiltered=None):
        self.fname = fname
        self.kind = kind
        self.itemsize = int(kind[1:])
        self.frame_size = frame_size
        self.frame_nbytes = frame_size * self.itemsize
        self.addr = addr
        self.nbytes = nbytes
        self.decoder = decoder
        self.unfiltered = unfiltered
        self.staging_dtype = _STAGING_DTYPE[kind]

    def describe(self):
        if self.decoder is None:
            return "uncompressed"
        return self.decoder.replace("bshuf_decompress_", "bitshuffle-")


def _chunk_table(data):
    """Return (addr, nbytes, filter_mask) per frame, or None if any chunk of
    a frame-per-chunk dataset is missing."""
    frame_num = data.shape[0]
    addr = np.full(frame_num, -1, dtype=np.int64)
    nbytes = np.zeros(frame_num, dtype=np.int64)
    filter_mask = np.zeros(frame_num, dtype=np.uint32)

    def store(info):
        n = info.chunk_offset[0]
        if info.byte_offset is not None:
            addr[n] = info.byte_offset
            nbytes[n] = info.size
            filter_mask[n] = info.filter_mask

    if hasattr(data.id, "chunk_iter"):
        data.id.chunk_iter(store)
    else:
        # HDF5 < 1.12.3
        for n in range(data.id.get_num_chunks()):
            store(data.id.get_chunk_info(n))

    if np.any(addr < 0):
        return None
    return addr, nbytes, filter_mask


def probe_direct_source(data):
    """Return a DirectSource if the h5py dataset `data` can be read without
    the HDF5 library, else None.

    Qualifying layouts are 3D datasets of a native fixed-size type that are
    either contiguous, or chunked one frame per chunk with no filter or with
    the bitshuffle (LZ4/ZSTD) filter alone.
    """
    kind = data.dtype.str[1:]
    if not (
        data.ndim == 3
        and data.dtype.isnative
        and kind in _STAGING_DTYPE
        and not data.is_virtual
        and data.file.driver == "sec2"
        and data.file.userblock_size == 0
        and data.shape[0] > 0
        and hasattr(os, "preadv")
    ):
        return None

    # the file that actually holds the data; differs from the opened file
    # when the dataset is reached through an external link
    fname = data.file.filename
    frame_size = data.shape[1] * data.shape[2]
    frame_nbytes = frame_size * data.dtype.itemsize

    if data.chunks is None:
        offset = data.id.get_offset()
        if offset is None or data.external is not None:
            return None
        addr = offset + frame_nbytes * np.arange(data.shape[0], dtype=np.int64)
        nbytes = np.full(data.shape[0], frame_nbytes, dtype=np.int64)
        return DirectSource(fname, kind, frame_size, addr, nbytes)

    if data.chunks != (1,) + data.shape[1:]:
        return None

    dcpl = data.id.get_create_plist()
    decoder = None
    if dcpl.get_nfilters() == 1:
        filter_id, _, cd_values, _ = dcpl.get_filter(0)
        # cd_values: (major, minor, elem_size, block_size, compression)
        if filter_id != BSHUF_FILTER_ID or len(cd_values) < 5:
            return None
        decoder = _BSHUF_DECODERS.get(cd_values[4])
        if decoder is None or cd_values[2] != data.dtype.itemsize:
            return None
        if _load_bshuf() is None:
            return None
    elif dcpl.get_nfilters() != 0:
        return None

    table = _chunk_table(data)
    if table is None:
        return None
    addr, nbytes, filter_mask = table
    # bit n of the mask set -> filter n was skipped when writing that chunk
    unfiltered = (filter_mask & 1).astype(bool) if decoder is not None else None
    return DirectSource(fname, kind, frame_size, addr, nbytes, decoder, unfiltered)


class FrameDecoder(object):
    """Read single frames of a DirectSource into caller-provided buffers.

    `decode` is thread safe and spends its time in pread and in the
    bitshuffle library, both of which release the GIL.
    """

    def __init__(self, source):
        self.source = source
        self.fd = os.open(source.fname, os.O_RDONLY)
        self._func = None
        if source.decoder is not None:
            self._func = getattr(_load_bshuf(), source.decoder)
        self._local = threading.local()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _pread(self, buf, offset):
        done = 0
        while done < buf.size:
            n = os.preadv(self.fd, [buf[done:]], offset + done)
            if n <= 0:
                raise IOError(f"unexpected end of file while reading {self.source.fname}")
            done += n

    def _scratch(self, name, nbytes):
        # per-thread buffers: the compressed chunk, the frame before cropping
        buf = getattr(self._local, name, None)
        if buf is None or buf.size < nbytes:
            buf = np.empty(nbytes + nbytes // 4, dtype=np.uint8)
            setattr(self._local, name, buf)
        return buf[:nbytes]

    def decode(self, frame, out, crop=None):
        """Fill the contiguous 1D array `out` with frame `frame`, or with its
        pixels `crop` (an index array) if given."""
        if crop is None:
            self._decode(frame, out)
        else:
            full = self._scratch("frame", self.source.frame_nbytes).view(out.dtype)
            self._decode(frame, full)
            # out is only written directly for modes other than "raise"; the
            # caller has checked the indices
            np.take(full, crop, out=out, mode="clip")

    def _decode(self, frame, out):
        src = self.source
        nbytes = int(src.nbytes[frame])
        offset = int(src.addr[frame])
        if self._func is None or src.unfiltered[frame]:
            if nbytes != src.frame_nbytes:
                raise ValueError(f"frame {frame}: unexpected raw chunk size {nbytes}")
            self._pread(out.view(np.uint8), offset)
            return

        buf = self._scratch("chunk", nbytes)
        self._pread(buf, offset)
        total, block_nbytes = _BSHUF_HEADER.unpack_from(buf)
        if total != src.frame_nbytes:
            raise ValueError(f"frame {frame}: chunk holds {total} bytes, expected {src.frame_nbytes}")
        ret = self._func(
            buf.ctypes.data + _BSHUF_HEADER.size,
            out.ctypes.data,
            src.frame_size,
            src.itemsize,
            block_nbytes // src.itemsize,
        )
        if ret < 0:
            raise RuntimeError(f"frame {frame}: bitshuffle decompression failed with error {ret}")


def cast_like_hdf5(x, kind, dtype):
    """Convert a staged batch of file dtype `kind` to torch `dtype`.

    Reproduces what HDF5 does when h5py reads into a buffer of another type:
    values are preserved, and those that do not fit saturate (uint32 -> int32
    clamps at 2^31 - 1, uint16 -> int16 at 2^15 - 1).
    """
    if kind == "u2":
        x = x.to(torch.int32) & 0xFFFF
        if dtype == torch.int16:
            x = x.clamp_(max=2**15 - 1)
    elif kind == "u4":
        if dtype == torch.int32:
            x = x.masked_fill(x < 0, 2**31 - 1)
        else:
            x = x.to(torch.int64) & 0xFFFFFFFF
    return x.to(dtype)


# The staging buffers of the most recently closed reader. Page-locking memory
# is slow (~0.5 s/GB), so consecutive datasets of the same geometry (batch
# jobs, segments) reuse them.
_staging_cache = {}


def acquire_staging(key):
    """Return `count` buffers for key = (count, shape, dtype, pinned)."""
    count, shape, dtype, pinned = key
    staging = _staging_cache.pop(key, None)
    _staging_cache.clear()
    if staging is None:
        staging = [torch.empty(shape, dtype=dtype, pin_memory=pinned) for _ in range(count)]
    return staging


def release_staging(key, staging):
    _staging_cache.clear()
    _staging_cache[key] = staging


class BatchPrefetcher(object):
    """Decode batches ahead of the consumer and hand them over on the device.

    `batches` lists the frame indices of each batch. A producer thread walks
    it in order: the frames of as many batches as there are staging buffers
    are decoding in `pool` at any time; each finished batch is copied to
    `device` and queued for `get`. With `crop`, an index array, only those
    pixels of each frame are kept.
    """

    def __init__(self, decoder, batches, staging, pool, device, crop=None, queue_depth=2):
        self._decoder = decoder
        self._batches = batches
        self._crop = crop
        self._staging = staging
        self._pool = pool
        self._device = torch.device(device)
        # copy on a stream of its own so that the transfers overlap with the
        # correlation kernels the consumer queues on its stream
        self._stream = None
        if self._device.type == "cuda":
            self._stream = torch.cuda.Stream(self._device)
        self._queue = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hdf-prefetch", daemon=True)
        self._thread.start()

    def _to_device(self, buf):
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                x = buf.to(self._device, non_blocking=True)
                # the staging buffer is reused as soon as this returns
                self._stream.synchronize()
            return x
        if self._device.type == "cpu":
            return buf.clone()
        return buf.to(self._device)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self):
        free = deque(self._staging)
        pending = deque()
        batches = iter(self._batches)
        try:
            while not self._stop.is_set():
                while free:
                    frames = next(batches, None)
                    if frames is None:
                        break
                    buf = free.popleft()
                    rows = buf.numpy()
                    futures = [
                        self._pool.submit(self._decoder.decode, frame, rows[n], self._crop)
                        for n, frame in enumerate(frames)
                    ]
                    pending.append((buf, futures))
                if not pending:
                    break
                buf, futures = pending[0]
                for future in futures:
                    future.result()
                x = self._to_device(buf[: len(futures)])
                pending.popleft()
                free.append(buf)
                self._put(x)
        except BaseException as e:
            self._put(e)
        finally:
            # nothing may write to the staging buffers once this thread ends
            for _, futures in pending:
                for future in futures:
                    if not future.cancel():
                        future.exception()

    def get(self):
        """Return the next batch, on the device."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError("the HDF5 prefetch thread stopped unexpectedly")
        if isinstance(item, BaseException):
            raise item
        if self._stream is not None:
            # the batch was allocated on the copy stream but is consumed on
            # this one: keep its memory from being reused for a later copy
            # until the kernels queued here have run
            item.record_stream(torch.cuda.current_stream(self._device))
        return item

    def stop(self):
        self._stop.set()
        while self._thread.is_alive():
            # make room for a producer that is blocked on a full queue
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=0.005)
