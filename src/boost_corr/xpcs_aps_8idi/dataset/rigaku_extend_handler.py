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

        all_index = np.arange(0, self.frame_num_raw + 2)
        all_index[-1] = self.frame_num_raw
        # build an index map for all indexes
        beg = np.searchsorted(frame, all_index[:-1])
        end = np.searchsorted(frame, all_index[1:])
        mem_addr = [slice(a, b) for a, b in zip(beg, end)]
        return (index, frame, count), mem_addr
