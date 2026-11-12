import logging
import os
import sys
import h5py
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)


key_map = {
    "dqmap": "/qmap/dynamic_roi_map",
    "sqmap": "/qmap/static_roi_map",
    "mask": "/qmap/mask",
    "blemish": "/qmap/blemish",
}


# each q-bin is accumulated in up to MAX_SUB_BINS sub-bins, and each image in
# up to MAX_SLOTS sub-bins in total
MAX_SUB_BINS = 128
MAX_SLOTS = 2**16


def get_sub_bin_num(size):
    return max(1, min(MAX_SUB_BINS, MAX_SLOTS // size))


def get_slot_index(qmap: torch.Tensor, size: int) -> torch.Tensor:
    """
    Converts a q-bin index map to the index of the sub-bins (slots) that the
    pixels are accumulated in. The pixels of a q-bin are spread over several
    slots, so that neighboring pixels, which mostly share the q-bin, are not
    added to the same memory location. It avoids the write contention of
    index_add_ on GPU, and the shorter float32 sums are more accurate.

    The slot index only depends on the qmap; compute it once and reuse it for
    all the datasets.
    """
    qmap = qmap.view(-1).long()
    pixel = torch.arange(qmap.shape[0], device=qmap.device)
    num_sub = get_sub_bin_num(size)
    return qmap * num_sub + pixel % num_sub


def sum_with_slot_index(img: torch.Tensor, slot: torch.Tensor, size: int):
    """
    Sums img of shape (C, N) over the q-bins for all the C images at once.
    Returns a tensor of shape (C, size); the first column is the index 0,
    i.e. the pixels that belong to no q-bin.
    """
    if img.dtype not in (torch.float32, torch.float64):
        img = img.float()
    num_sub = get_sub_bin_num(size)
    total = torch.zeros(
        img.shape[0], size * num_sub, dtype=img.dtype, device=img.device
    )
    total.index_add_(1, slot, img)
    return total.view(img.shape[0], size, num_sub).sum(dim=2)


def find_bin_count(qmap, minlength=None):
    count = np.bincount(qmap.ravel(), minlength=minlength)[1:]
    count = count.astype(np.int32)
    nan_idx = count == 0
    count[nan_idx] = -1
    return count, nan_idx


def average_with_index_map(
    img: torch.Tensor,
    qmap: torch.Tensor,
    size: Optional[int] = None,
    norm_factor: Optional[torch.Tensor] = None,
    slot: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the cluster-wise average intensity of an image tensor based on a
    quantization/segmentation map.

    Parameters
    ----------
    img : torch.Tensor
        A tensor of shape (C, H, W) or (C, N) where C is the number of images
        (or channels), and H, W represent the height and width of the image.
        The tensor contains pixel intensities.
    qmap : torch.Tensor
        A tensor of shape (H, W) or (N,) containing cluster labels
        (quantization indices) for each pixel.
    size : int, optional
        The total number of clusters (quantization levels). If not provided,
        it is inferred from `qmap`.
    norm_factor : torch.Tensor, optional
        A precomputed normalization factor for each cluster. If None, it is
        computed based on `qmap`.
    slot : torch.Tensor, optional
        The precomputed `get_slot_index(qmap, size)`. If None, it is computed
        based on `qmap`.

    Returns
    -------
    torch.Tensor
        A tensor of shape (C, size-1) where each value represents the average
        intensity of the corresponding cluster.

    Notes
    -----
    - The function assumes `qmap` contains indices starting from 1
      (i.e., it ignores index `0`).
    - Clusters with zero pixels are handled by setting their normalization
      factor to `1` to avoid division by zero.
    """
    qmap = qmap.view(-1)
    num_pixels = qmap.shape[0]

    if size is None:
        size = qmap.max().item() + 1  # Infer size from qmap if not provided

    if norm_factor is None:
        norm_factor = torch.bincount(qmap, minlength=size)[1:]
        norm_factor[norm_factor == 0] = 1  # Avoid division by zero

    if slot is None:
        slot = get_slot_index(qmap, size)

    img = img.reshape(-1, num_pixels)
    summed_values = sum_with_slot_index(img, slot, size)[:, 1:]

    avg = summed_values.float() / norm_factor
    return avg


def convert_to_numpy(output_dict):
    for k, v in output_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.float().cpu().numpy()
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            output_dict[k] = v.astype(np.float32)
    return output_dict


class XpcsQPartitionMap(object):
    def __init__(
        self,
        qmap_fname,
        flag_fix=False,
        flag_sort=False,
        dq_selection=None,
        crop_ratio_threshold=0.5,
        device="cpu",
    ) -> None:
        super().__init__()
        self.fname = qmap_fname
        self.device = device
        self.masked_ratio = 1.0
        self.masked_pixels = 0
        self.dq_dim = 0
        self.sq_dim = 0
        self.dqmap = None
        self.sqmap = None
        self.mask = None
        self.det_size = None
        self.qinfo = None
        self.flag_sort = flag_sort
        self.load(flag_fix)
        self.group_qmap(dq_selection)
        self.mask_crop = self.update_mask_crop(crop_ratio_threshold)

    def group_qmap(self, dq_selection=None):
        scount, snan_idx = find_bin_count(self.sqmap, self.sq_dim + 1)
        if self.flag_sort:
            # select and sort the qmap; good for twotime and multitau
            mask_idx_1d, qinfo = self.select_sort(dq_selection)
        else:
            # don't sort the qmap; good for multitau
            mask_idx_1d = np.where(self.mask.reshape(-1) == 1)[0]
            qinfo = None

        info_np = {
            "mask_idx_1d": mask_idx_1d,
            "sqmap_full": self.sqmap.reshape(-1),
            "dqmap_crop": self.dqmap.reshape(-1)[mask_idx_1d],
            "sqmap_crop": self.sqmap.reshape(-1)[mask_idx_1d],
            "scount": scount,
            "snan_idx": snan_idx,
        }

        info = {}
        for k, v in info_np.items():
            info[k] = torch.tensor(v).to(self.device)

        self.info_np = info_np
        self.info = info
        self.qinfo = qinfo
        # derived from the maps above on first use; shared by all the datasets
        self.slot = {}
        self.g2_plan = None

    def get_slot(self, key, size):
        if key not in self.slot:
            self.slot[key] = get_slot_index(self.info[key], size)
        return self.slot[key]

    def get_g2_plan(self):
        """
        the part of compute_g2 that only depends on the qmap
        """
        if self.g2_plan is not None:
            return self.g2_plan

        dqmap = self.info["dqmap_crop"].long()
        sqmap = self.info["sqmap_crop"]
        num_dq, num_sq = self.dq_dim + 1, self.sq_dim + 1
        # number of pixels in each (dq, sq) pair; index 0 is not a q-bin
        pair = torch.bincount(dqmap * num_sq + sqmap, minlength=num_dq * num_sq)
        pair = pair.reshape(num_dq, num_sq).cpu().numpy()

        # the sq bins that overlap with each dq bin
        dq_idx, sq_idx = np.nonzero(pair[1:, 1:])
        sq_num = np.bincount(dq_idx, minlength=self.dq_dim)

        # the mean and the variance of the mean of the pixels in a dq bin
        count = pair[1:].sum(axis=1).astype(np.float64)
        with np.errstate(divide="ignore"):
            # a single pixel gives nan, the same as torch.std
            var_norm = 1.0 / ((count - 1) * count)
        var_norm[count == 0] = 0

        plan = {
            "dq_idx": dq_idx,
            "sq_idx": sq_idx,
            "sq_norm": (1.0 / np.maximum(sq_num, 1)).astype(np.float32),
            "mean_norm": (1.0 / np.maximum(count, 1)).astype(np.float32),
            "var_norm": var_norm.astype(np.float32),
        }
        for k, v in plan.items():
            plan[k] = torch.tensor(v, device=self.device)
        self.g2_plan = plan
        return plan

    def describe(self):
        logger.info(f"qmap file: {self.fname}")
        logger.info(f"flag_sort: {self.flag_sort}")
        logger.info(
            f"masked area pixels/ratio: "
            f"{self.masked_pixels}/"
            f"{self.masked_ratio * 100:.4f}%"
        )
        # omit dq = 0 which is not used
        dq_num = np.count_nonzero(np.bincount(self.dqmap.ravel())[1:])
        logger.info(f"dq_map length: {dq_num}")

    def load(self, flag_fix=False):
        values = {}
        with h5py.File(self.fname, "r") as f:
            for key, real_key in key_map.items():
                if real_key in f:
                    values[key] = np.squeeze(f[real_key][()])

        missing = [key_map[k] for k in ("dqmap", "sqmap", "mask") if k not in values]
        if missing:
            raise KeyError(f"{missing} not found in qmap file [{self.fname}]")
        # for backward compatibility; older qmap files do not have blemish
        if "blemish" not in values:
            values["blemish"] = np.copy(values["mask"])
            logger.info("no blemish data is available, using mask instead")
        else:
            logger.info("blemish data is read from qmap successfully")

        self.mask = values["mask"].astype(bool)
        self.blemish = values["blemish"].astype(bool)
        self.dqmap = values["dqmap"].astype(np.int32)
        self.sqmap = values["sqmap"].astype(np.int32)
        if flag_fix:
            self.check_fix_qmap()
        self.det_size = self.mask.shape
        self.dq_dim = int(np.max(self.dqmap))
        self.sq_dim = int(np.max(self.sqmap))
        self.masked_pixels = int(np.sum(self.mask))
        self.masked_ratio = self.masked_pixels / self.mask.size

    def update_file(self):
        logger.warning(f"update fixed qmap in file: [{self.fname}]")
        # update the qmap after fixing sqmap/dqmap inconsistency
        with h5py.File(self.fname, "a") as f:
            del f[key_map["dqmap"]]
            f[key_map["dqmap"]] = self.dqmap.astype(np.uint32)

    def update_rotation(self, det_size):
        if self.dqmap.shape != det_size:
            logger.warning("qmap is rotated 90 deg to match the detctor oriention.")
            self.dqmap = np.swapaxes(self.dqmap, 0, 1)
            self.sqmap = np.swapaxes(self.sqmap, 0, 1)
            self.mask = np.swapaxes(self.mask, 0, 1)
            self.det_size = self.mask.shape
            assert self.det_size == det_size, (
                "The shape of QMap does not match the raw data shape"
            )
            # need redo the preprocessing
            self.group_qmap()
            return True
        return False

    def update_mask_crop(self, crop_ratio_threshold):
        if self.masked_ratio < crop_ratio_threshold:
            logger.info("masked_ratio is too low. Apply crop-mask on the raw input.")
            mask_crop = self.info["mask_idx_1d"]
        else:
            mask_crop = None
        return mask_crop

    def check_fix_qmap(self):
        """
        make each sq bin belong to one dq bin, by moving its pixels to the dq
        bin that has most of them; returns True if nothing has to be fixed
        """
        num_dq, num_sq = int(self.dqmap.max()) + 1, int(self.sqmap.max()) + 1
        key = self.sqmap.ravel().astype(np.int64) * num_dq + self.dqmap.ravel()
        # number of pixels in each (sq, dq) pair; sq = 0 is not a bin
        pair = np.bincount(key, minlength=num_sq * num_dq).reshape(num_sq, num_dq)
        dq_num = np.count_nonzero(pair[1:], axis=1)
        assert np.all(dq_num <= 2), "only support sqmap belongs up to 2 bins"

        sq_list = np.flatnonzero(dq_num == 2) + 1
        for sq in sq_list:
            dq = np.flatnonzero(pair[sq])
            logger.warning(f"qmap inconsistent at sq: {sq} <-> {dq}: dq")
        if sq_list.size > 0:
            # the lower dq bin takes the pixels if both have the same number
            major_dq = np.argmax(pair, axis=1)
            roi = np.isin(self.sqmap, sq_list)
            self.dqmap[roi] = major_dq[self.sqmap[roi]]
        return sq_list.size == 0

    def select_sort(self, dq_selection=None):
        """
        group the sqmap and dqmap so that the pixels are continuous in memory
        """
        dqmap = self.dqmap.ravel()
        sqmap = self.sqmap.ravel()
        num_dq, num_sq = self.dq_dim + 1, self.sq_dim + 1
        dq_count = np.bincount(dqmap, minlength=num_dq)
        sq_count = np.bincount(sqmap, minlength=num_sq)

        if dq_selection is None:
            dq_selection = np.flatnonzero(dq_count)[1:].astype(dqmap.dtype)
        dq_selection = [dq for dq in sorted(set(dq_selection)) if 0 <= dq < num_dq]

        # sort the selected pixels by (dq, sq); the stable sort keeps the
        # pixels of a sq bin in the detector order
        selected = np.zeros(num_dq, dtype=bool)
        selected[dq_selection] = True
        pixel_idx = np.flatnonzero(selected[dqmap])
        key = dqmap[pixel_idx].astype(np.int64) * num_sq + sqmap[pixel_idx]
        # number of pixels in each (dq, sq) pair
        pair = np.bincount(key, minlength=num_dq * num_sq).reshape(num_dq, num_sq)
        if num_dq * num_sq <= 2**16:
            # numpy only has the radix sort for 16-bit integers
            key = key.astype(np.uint16)
        mask_idx_1d_sort = pixel_idx[np.argsort(key, kind="stable")]

        info = {"dq_slc": [], "sq_slc": [], "dq_idx": [], "sq_idx": [], "dq_sq_map": {}}

        start = 0
        for dq in dq_selection:
            if dq_count[dq] == 0:
                continue
            info["dq_sq_map"][dq] = []
            info["dq_idx"].append(dq)
            info["dq_slc"].append(slice(start, start + int(dq_count[dq])))
            for sq in np.flatnonzero(pair[dq]).astype(sqmap.dtype):
                # a sq bin must be inside of one dq bin
                assert pair[dq, sq] == sq_count[sq], "dqmap and sqmap must be consistent"
                stop = start + int(sq_count[sq])
                info["sq_slc"].append(slice(start, stop))
                info["sq_idx"].append(sq)
                info["dq_sq_map"][dq].append(sq)
                start = stop

        dq_idx = info["dq_idx"]
        total_dq, dq_min, dq_max = len(dq_idx), min(dq_idx), max(dq_idx)
        logger.info(f"total dq: {total_dq}. min - max: {dq_min} - {dq_max}")
        return mask_idx_1d_sort, info

    def normalize_sqmap(self, img, flag_crop, apply_nan=True):
        scount = self.info["scount"]
        key = "sqmap_crop" if flag_crop else "sqmap_full"
        size = self.sq_dim + 1
        slot = self.get_slot(key, size)
        result = average_with_index_map(img, self.info[key], size, scount, slot)

        if apply_nan:
            snan_idx = self.info["snan_idx"]
            result = result[..., ~snan_idx]
        return result

    def recover_dimension(self, saxs2d, flag_crop, apply_mask=False):
        if flag_crop:
            pixel_num = self.det_size[0] * self.det_size[1]
            full_img = torch.zeros(pixel_num, dtype=torch.float32, device=saxs2d.device)
            full_img[self.info["mask_idx_1d"]] = saxs2d
        else:
            full_img = saxs2d

        full_img = full_img.reshape(1, *self.det_size)
        device = saxs2d.device

        if not apply_mask:
            full_img = full_img * torch.from_numpy(self.blemish).to(device)
        else:
            full_img = full_img * torch.from_numpy(self.mask).to(device)

        return full_img

    def normalize_multitau(self, res, save_G2=False):
        flag_crop = res["mask_crop"] is not None
        g2, g2_err = self.compute_g2(res["G2"], flag_crop)
        output_dir = {"delay_list": res["tau"], "g2": g2, "g2_err": g2_err}

        if save_G2:
            G2 = res["G2"].reshape(-1, res["G2"].shape[-1])
            if flag_crop:
                pixel_num = self.det_size[0] * self.det_size[1]
                value = torch.zeros(
                    G2.shape[0], pixel_num, dtype=torch.float32, device=G2.device
                )
                value[:, self.info["mask_idx_1d"]] = G2
            else:
                value = G2

            # the final shape of G2IPIF is (num_tau, 3, det_row, det_col)
            output_dir["G2IPIF"] = value.reshape(-1, 3, *self.det_size)
        output_dir = convert_to_numpy(output_dir)
        return output_dir

    def normalize_scattering(self, res):
        flag_crop = res["mask_crop"] is not None
        if res["saxs_2d"].shape == self.det_size:
            flag_crop = False
        saxs1d = self.normalize_sqmap(res["saxs_2d"], flag_crop)
        saxs1d_par = self.normalize_sqmap(res["saxs_2d_par"], flag_crop)
        saxs2d = self.recover_dimension(res["saxs_2d"], flag_crop, apply_mask=False)
        output_dir = {
            "saxs_2d": saxs2d,
            "saxs_1d": saxs1d,
            "saxs_1d_segments": saxs1d_par,
            "intensity_vs_time": res["intensity_vs_time"],
        }
        output_dir = convert_to_numpy(output_dir)
        return output_dir

    def compute_g2(self, G2, flag_crop):
        if not flag_crop:
            G2 = G2[..., self.info["mask_idx_1d"]]
        flag_crop = True
        plan = self.get_g2_plan()

        G2_sq = self.normalize_sqmap(G2, flag_crop, apply_nan=False)
        G2_sq = G2_sq.reshape(-1, 3, G2_sq.shape[1])

        IP_IF_sq = G2_sq[:, 1] * G2_sq[:, 2]
        IP_IF_sq[IP_IF_sq == 0] = 1.0e8
        g2_sq = G2_sq[:, 0] / IP_IF_sq
        # average the sq bins that belong to each dq bin
        g2 = torch.zeros(G2.shape[0], self.dq_dim, device=G2.device)
        g2.index_add_(1, plan["dq_idx"], g2_sq[:, plan["sq_idx"]])
        g2 *= plan["sq_norm"]

        IP_IF = G2[:, 1] * G2[:, 2]
        IP_IF[IP_IF == 0] = 1.0e8
        g2_pixel = torch.div(G2[:, 0], IP_IF, out=IP_IF)

        # standard error of the pixels in each dq bin; subtract the mean of
        # the bin first, as torch.std does, to keep the float32 precision
        size = self.dq_dim + 1
        slot = self.get_slot("dqmap_crop", size)
        mean = sum_with_slot_index(g2_pixel, slot, size)
        mean[:, 1:] *= plan["mean_norm"]
        g2_pixel -= mean[:, self.info["dqmap_crop"]]
        g2_pixel.square_()
        g2_var = sum_with_slot_index(g2_pixel, slot, size)[:, 1:] * plan["var_norm"]
        g2_err = torch.sqrt(g2_var)

        return g2, g2_err


def check_and_fix_qmap(fname):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    qpm = XpcsQPartitionMap(fname, flag_fix=False)
    flag = qpm.check_fix_qmap()
    if not flag:
        choise = input("update the qmap file? [Y/n] ")
        if choise == "Y":
            qpm.update_file()
    else:
        _, info = qpm.select_sort()
        for k, v in info["dq_sq_map"].items():
            logger.info(f"dq: {k} <-> {v}: sq")
        logger.info(f"[{fname}] is consistent")


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        check_and_fix_qmap(sys.argv[1])
    else:
        print("Usage: check_fix_qmap some_qmap.hdf")
