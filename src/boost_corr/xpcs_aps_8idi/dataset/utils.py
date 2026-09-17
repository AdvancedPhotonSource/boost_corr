import glob
import logging
import os

import h5py
import numpy as np

from .imm_handler import ImmDataset
from .rigaku_handler import RigakuDataset
from .rigaku_3M_handler import Rigaku3MDataset
from .rigaku_extend_handler import RigakuExtendDataset
from .hdf_handler import HdfDataset


logger = logging.getLogger(__name__)


def resolve_bin_width_ns(raw_fname: str, bin_time_s: float) -> float:
    """Resolve the physical time-bin width (ns) for a Rigaku-family sparse
    file's "frame" field, which has no self-describing bin width.

    Prefers the sibling `*_metadata.hdf`'s
    `entry/instrument/detector_1/frame_time` (seconds); falls back to
    `bin_time_s` (also seconds, e.g. from --bin-time-s) if no metadata file
    or field is found. Logs a warning if both are present and disagree.
    """
    fallback_ns = bin_time_s * 1e9
    meta_dir = os.path.dirname(os.path.abspath(raw_fname))
    for meta_fname in sorted(glob.glob(os.path.join(meta_dir, "*_metadata.hdf"))):
        try:
            with h5py.File(meta_fname, "r") as f:
                path = "/entry/instrument/detector_1/frame_time"
                if path not in f:
                    continue
                meta_ns = float(f[path][()]) * 1e9
        except OSError:
            continue
        if not np.isclose(meta_ns, fallback_ns, rtol=1e-3):
            logger.warning(
                f"bin width mismatch: metadata {meta_fname} says {meta_ns:.3f} ns, "
                f"--bin-time-s says {fallback_ns:.3f} ns; using metadata value"
            )
        return meta_ns
    return fallback_ns


def create_dataset(raw_fname, **kwargs):
    if not os.path.isfile(raw_fname):
        raise FileNotFoundError(f"The raw_file '{raw_fname}' does not exist.")

    ext = os.path.splitext(raw_fname)[-1]
    # use_loader is set for HDF files, it can use multiple processes to read
    # large HDF file;
    use_loader = False
    if ext == ".bin":
        dataset_method = RigakuDataset
        use_loader = False
        batch_size = 1024
    elif raw_fname.endswith(".bin.000"):
        dataset_method = Rigaku3MDataset
        use_loader = False
        batch_size = 256
    elif ext == ".bix":
        dataset_method = RigakuExtendDataset
        use_loader = False
        batch_size = 1024
    elif ext in [".imm", ".h5", ".hdf", ".hdf5"]:
        # Check if it's an HDF5 file by trying to detect HDF5 format
        # If not HDF5, treat it as IMM format
        real_raw = os.path.realpath(raw_fname)
        try:
            is_hdf5 = h5py.is_hdf5(real_raw)
        except (OSError, IOError):
            # File might be corrupted or inaccessible
            is_hdf5 = False

        if is_hdf5:
            dataset_method = HdfDataset
            use_loader = True
            batch_size = 8
        else:
            dataset_method = ImmDataset
            use_loader = False
            batch_size = 8
    elif raw_fname.endswith((".tpx", ".tpx.000", ".tpx.001", ".tpx.002")):
        from .timepix4_handler import TimepixAnalysisDataset

        dataset_method = TimepixAnalysisDataset
        use_loader = False
        batch_size = 1024
    else:
        raise TypeError(f"File type [{ext}] is not supported")

    dset = dataset_method(raw_fname, batch_size=batch_size, **kwargs)

    return dset, use_loader
