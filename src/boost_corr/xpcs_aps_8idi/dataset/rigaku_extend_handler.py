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

    def __init__(self, *args, **kwargs):
        super(RigakuExtendDataset, self).__init__(*args, **kwargs)
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
