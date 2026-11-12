import numpy as np
from .rigaku_handler import RigakuDataset
from .rigaku_utils import convert_sparse_extend


class RigakuExtendDataset(RigakuDataset):
    """
    Parameters
    ----------
    filename: string
        path to .bix file
    """

    def __init__(self, *args, det_size=(512, 448), **kwargs):
        # (512, 448) matches the Timepix4/Tempus single-chip module shape
        # (ROWS=512, COLS=448 in the timepix_dataset package's
        # alexandr_version/settings.py) that this format's raw pixel-index
        # encoding actually uses -- NOT the generic (512, 1024) XpcsDataset
        # default RigakuDataset inherits, which is sized for real Rigaku
        # hardware. A qmap or dataset built assuming the wrong width silently
        # points ring/ROI pixel indices at the wrong (row, col) locations
        # (they don't error -- they just correlate the wrong pixels, or a
        # gapped qmap crash comes for free from the shape-mismatch check).
        # Still overridable via an explicit det_size= kwarg.
        super(RigakuExtendDataset, self).__init__(*args, det_size=det_size, **kwargs)
        self.dataset_type = "RigakuExtend 64bit Binary"
        # .bix files are, in practice, only ever used with the event
        # correlator (that's the whole reason this format's "frame" field is
        # a fine time bin rather than a real camera frame) -- get_events()
        # never calls __getbatch__/get_sparsity(), so skip the per-frame
        # sparsity check in describe(): it forces building the lazy
        # mem_addr table for no benefit, and "fraction of pixels hit in one
        # 10ns-scale time bin" isn't a meaningful density metric anyway.
        self.skip_sparsity_check = True

    def read_data(self, total_frames=None):
        with open(self.fname, "r") as f:
            a = np.fromfile(f, dtype=np.uint64)
            d = convert_sparse_extend(a)
            index, frame, count = d[0], d[1], d[2]
            if total_frames is None:
                total_frames = frame[-1] + 1
            else:
                total_frames = max(total_frames, frame[-1] + 1)

        # update frame_num and batch_num
        self.update_batch_info(total_frames)
        # mem_addr is built lazily -- see RigakuDataset.mem_addr's docstring.
        self._frame_for_mem_addr = frame
        return (index, frame, count)
