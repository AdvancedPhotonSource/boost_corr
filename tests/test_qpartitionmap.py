#!/usr/bin/env python
"""XpcsQPartitionMap: the vectorized G2 -> g2 reduction and the sorted pixel
layout must reproduce the straightforward per-bin loops they replaced."""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np
import torch

from boost_corr.xpcs_aps_8idi.xpcs_qpartitionmap import (
    XpcsQPartitionMap,
    average_with_index_map,
)

DET_SIZE = (48, 64)
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def make_maps(sq_per_dq=4, num_dq=5):
    """rings for the dq bins; each ring is split into angular sq bins"""
    y, x = np.indices(DET_SIZE)
    r = np.hypot(y - DET_SIZE[0] / 2, x - DET_SIZE[1] / 2)
    phi = np.arctan2(y - DET_SIZE[0] / 2, x - DET_SIZE[1] / 2)
    dqmap = np.clip(r // 5, 0, num_dq).astype(np.uint32)  # the center is 0
    sector = ((phi + np.pi) / (2 * np.pi) * sq_per_dq).astype(np.uint32)
    sector = np.clip(sector, 0, sq_per_dq - 1)
    sqmap = np.where(dqmap > 0, (dqmap - 1) * sq_per_dq + sector + 1, 0)
    return dqmap, sqmap.astype(np.uint32), (dqmap > 0).astype(np.uint32)


def reference_g2(G2, dqmap, sqmap, mask):
    """the per-bin loops, in float64; G2 has the full detector shape"""
    dq_dim, sq_dim = int(dqmap.max()), int(sqmap.max())
    idx = np.flatnonzero(mask.ravel())
    dq, sq = dqmap.ravel()[idx].astype(int), sqmap.ravel()[idx].astype(int)
    G2 = G2.reshape(G2.shape[0], 3, -1)[..., idx].astype(np.float64)

    scount = np.bincount(sqmap.ravel(), minlength=sq_dim + 1)[1:].astype(float)
    scount[scount == 0] = -1
    G2_sq = np.stack(
        [
            np.bincount(sq, weights=x, minlength=sq_dim + 1)[1:]
            for x in G2.reshape(-1, G2.shape[-1])
        ]
    )
    G2_sq = (G2_sq / scount).reshape(-1, 3, sq_dim)
    IP_IF_sq = G2_sq[:, 1] * G2_sq[:, 2]
    IP_IF_sq[IP_IF_sq == 0] = 1.0e8
    g2_sq = G2_sq[:, 0] / IP_IF_sq

    IP_IF = G2[:, 1] * G2[:, 2]
    IP_IF[IP_IF == 0] = 1.0e8
    g2_pixel = G2[:, 0] / IP_IF

    g2 = np.zeros((G2.shape[0], dq_dim))
    g2_err = np.zeros_like(g2)
    for n in range(1, dq_dim + 1):
        roi = dq == n
        if roi.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            g2_err[:, n - 1] = np.std(g2_pixel[:, roi], axis=1, ddof=1)
        g2_err[:, n - 1] /= np.sqrt(roi.sum())
        roi_sq = np.unique(sq[roi])
        roi_sq = roi_sq[roi_sq > 0] - 1
        if roi_sq.size > 0:
            g2[:, n - 1] = g2_sq[:, roi_sq].mean(axis=1)
    return g2, g2_err


def reference_select_sort(dqmap, sqmap, dq_selection=None):
    """the per-bin loops that select_sort replaced"""
    dqmap, sqmap = dqmap.ravel(), sqmap.ravel()
    if dq_selection is None:
        dq_selection = np.unique(dqmap[dqmap > 0])
    info = {"dq_slc": [], "sq_slc": [], "dq_idx": [], "sq_idx": [], "dq_sq_map": {}}
    q_roi = []
    dq_start = 0
    for dq in sorted(dq_selection):
        roi = np.nonzero(dqmap == dq)[0]
        if len(roi) == 0:
            continue
        info["dq_sq_map"][dq] = []
        sq_start = dq_start
        for sq in np.sort(np.unique(sqmap[roi])):
            sq_roi = np.nonzero(sqmap == sq)[0]
            q_roi.append(sq_roi)
            info["sq_slc"].append(slice(sq_start, sq_start + len(sq_roi)))
            info["sq_idx"].append(sq)
            info["dq_sq_map"][dq].append(sq)
            sq_start += len(sq_roi)
        info["dq_idx"].append(dq)
        info["dq_slc"].append(slice(dq_start, dq_start + len(roi)))
        dq_start += len(roi)
        assert dq_start == sq_start, "dqmap and sqmap must be consistent"
    return np.hstack(q_roi), info


class QMapTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write_qmap(self, dqmap, sqmap, mask):
        fname = os.path.join(self.tmpdir, "qmap.h5")
        with h5py.File(fname, "w") as f:
            f["/qmap/dynamic_roi_map"] = dqmap
            f["/qmap/static_roi_map"] = sqmap
            f["/qmap/mask"] = mask
        return fname

    def random_G2(self, num_tau=6):
        """G2, IP, IF of a noisy g2 that depends on the pixel"""
        num_pix = DET_SIZE[0] * DET_SIZE[1]
        IP = self.rng.uniform(0.5, 2.0, (num_tau, num_pix))
        IF = self.rng.uniform(0.5, 2.0, (num_tau, num_pix))
        G2 = IP * IF * self.rng.uniform(1.0, 1.4, (num_tau, num_pix))
        return np.stack([G2, IP, IF], axis=1).astype(np.float32)


class TestComputeG2(QMapTestCase):
    def check(self, dqmap, sqmap, mask, G2=None, sortable=True):
        fname = self.write_qmap(dqmap, sqmap, mask)
        if G2 is None:
            G2 = self.random_G2()
        ref_g2, ref_err = reference_g2(G2, dqmap, sqmap, mask)
        for device in DEVICES:
            for flag_sort in (False, True):
                # the sorted layout takes dq > 0, not the mask
                if flag_sort and not (sortable and np.all((dqmap > 0) == (mask > 0))):
                    continue
                with self.subTest(device=device, flag_sort=flag_sort):
                    qpm = XpcsQPartitionMap(fname, device=device, flag_sort=flag_sort)
                    full = torch.tensor(G2, device=device)
                    crop = full[..., qpm.info["mask_idx_1d"]]
                    # twice: the second call reuses what the first precomputed
                    for flag_crop, x in [(False, full), (True, crop)] * 2:
                        g2, g2_err = qpm.compute_g2(x, flag_crop)
                        self.assertEqual(g2.dtype, torch.float32)
                        np.testing.assert_allclose(
                            g2.cpu().numpy(), ref_g2, rtol=2e-5, atol=1e-7
                        )
                        np.testing.assert_allclose(
                            g2_err.cpu().numpy(), ref_err, rtol=2e-4, atol=1e-7
                        )
        return ref_g2, ref_err

    def test_consistent_qmap(self):
        ref_g2, ref_err = self.check(*make_maps())
        self.assertTrue(np.all(ref_g2 > 1) and np.all(ref_err > 0))

    def test_mask_smaller_than_qmap(self):
        dqmap, sqmap, mask = make_maps()
        mask[:, :20] = 0
        self.check(dqmap, sqmap, mask)

    def test_masked_pixels_without_q_bin(self):
        dqmap, sqmap, mask = make_maps()
        dqmap[10:14, 10:30] = 0  # in the mask and a sq bin, but in no dq bin
        sqmap[30:34, 10:30] = 0  # in the mask and a dq bin, but in no sq bin
        self.check(dqmap, sqmap, mask)

    def test_empty_and_single_pixel_dq_bins(self):
        dqmap, sqmap, mask = make_maps()
        mask[dqmap == 2] = 0  # dq 2 has no pixels left
        keep = np.flatnonzero(dqmap.ravel() == 4)[0]
        mask[dqmap == 4] = 0
        mask.ravel()[keep] = 1  # dq 4 has a single pixel left
        ref_g2, ref_err = self.check(dqmap, sqmap, mask)
        self.assertTrue(np.all(ref_g2[:, 1] == 0) and np.all(ref_err[:, 1] == 0))
        self.assertTrue(np.all(np.isnan(ref_err[:, 3])))

    def test_sq_bin_in_two_dq_bins(self):
        dqmap, sqmap, mask = make_maps()
        sqmap[sqmap == 6] = 2  # sq 2 now overlaps with dq 1 and dq 2
        self.check(dqmap, sqmap, mask, sortable=False)

    def test_zero_intensity(self):
        dqmap, sqmap, mask = make_maps()
        G2 = self.random_G2()
        G2[:, :, sqmap.ravel() == 3] = 0  # a dark sq bin
        G2[:, :, ::7] = 0  # and dark pixels everywhere
        self.check(dqmap, sqmap, mask, G2)

    def test_uniform_pixels_have_zero_error(self):
        dqmap, sqmap, mask = make_maps()
        G2 = np.ones((4, 3, DET_SIZE[0] * DET_SIZE[1]), dtype=np.float32)
        G2[:, 0] = 1.3
        ref_g2, ref_err = self.check(dqmap, sqmap, mask, G2)
        np.testing.assert_allclose(ref_g2, 1.3)
        np.testing.assert_allclose(ref_err, 0, atol=1e-7)

    def test_fine_qmap_uses_fewer_sub_bins(self):
        # more q-bins than MAX_SLOTS / MAX_SUB_BINS
        dqmap, sqmap, mask = make_maps(sq_per_dq=150)
        self.assertGreater(sqmap.max() + 1, 2**16 // 128)
        self.check(dqmap, sqmap, mask)

    def test_regroup_drops_the_precomputed_values(self):
        qpm = XpcsQPartitionMap(self.write_qmap(*make_maps()))
        G2 = torch.tensor(self.random_G2())
        before = qpm.compute_g2(G2, False)
        self.assertTrue(qpm.slot and qpm.g2_plan is not None)
        qpm.group_qmap()
        self.assertTrue(not qpm.slot and qpm.g2_plan is None)
        after = qpm.compute_g2(G2, False)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a.numpy(), b.numpy())


class TestAverageWithIndexMap(QMapTestCase):
    def test_matches_bincount(self):
        _, sqmap, _ = make_maps()
        size = int(sqmap.max()) + 1
        img = self.rng.uniform(0, 100, (5, sqmap.size))
        count = np.bincount(sqmap.ravel(), minlength=size)[1:]
        ref = np.stack(
            [np.bincount(sqmap.ravel(), weights=x, minlength=size)[1:] for x in img]
        )
        ref = ref / count
        for device in DEVICES:
            qmap = torch.tensor(sqmap.astype(np.int32), device=device)
            for dtype in (torch.float32, torch.float64, torch.int32):
                with self.subTest(device=device, dtype=dtype):
                    x = torch.tensor(img, device=device).to(dtype)
                    expected = ref
                    if dtype == torch.int32:
                        expected = np.stack(
                            [
                                np.bincount(sqmap.ravel(), weights=v, minlength=size)
                                for v in np.floor(img)
                            ]
                        )[:, 1:] / count
                    out = average_with_index_map(x, qmap)
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertEqual(tuple(out.shape), (5, size - 1))
                    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-5)
                    # a 2d image and an explicit size give the same answer
                    out2 = average_with_index_map(
                        x.reshape(5, *DET_SIZE), qmap.reshape(DET_SIZE), size
                    )
                    # not bitwise: the order of the sums on GPU is not fixed
                    np.testing.assert_allclose(
                        out.cpu().numpy(), out2.cpu().numpy(), rtol=1e-6
                    )


class TestSelectSort(QMapTestCase):
    def check(self, dqmap, sqmap, mask, dq_selection=None):
        qpm = XpcsQPartitionMap(self.write_qmap(dqmap, sqmap, mask))
        selection = None if dq_selection is None else list(dq_selection)
        ref_idx, ref_info = reference_select_sort(qpm.dqmap, qpm.sqmap, selection)
        idx, info = qpm.select_sort(dq_selection)
        np.testing.assert_array_equal(idx, ref_idx)
        for key in ("dq_slc", "sq_slc", "dq_idx", "sq_idx"):
            self.assertEqual(list(info[key]), list(ref_info[key]), key)
        self.assertEqual(
            {k: list(v) for k, v in info["dq_sq_map"].items()},
            {k: list(v) for k, v in ref_info["dq_sq_map"].items()},
        )
        return info

    def test_all_bins(self):
        info = self.check(*make_maps())
        # the bin labels keep the dtype of the maps; they go to the result file
        self.assertEqual(np.asarray(info["dq_idx"]).dtype, np.int32)
        self.assertEqual(np.asarray(info["sq_idx"]).dtype, np.int32)

    def test_selection(self):
        self.check(*make_maps(), dq_selection=[4, 2])
        # a bin that is not in the map is skipped
        info = self.check(*make_maps(), dq_selection=[1, 3, 99])
        self.assertEqual(info["dq_idx"], [1, 3])

    def test_missing_dq_bin(self):
        dqmap, sqmap, mask = make_maps()
        sqmap[dqmap == 3] = 0
        mask[dqmap == 3] = 0
        dqmap[dqmap == 3] = 0
        info = self.check(dqmap, sqmap, mask)
        self.assertEqual(list(info["dq_idx"]), [1, 2, 4, 5])

    def test_more_bins_than_uint16(self):
        self.check(*make_maps(sq_per_dq=150, num_dq=600))

    def test_inconsistent_qmap(self):
        dqmap, sqmap, mask = make_maps()
        sqmap[sqmap == 6] = 2  # sq 2 now overlaps with dq 1 and dq 2
        qpm = XpcsQPartitionMap(self.write_qmap(dqmap, sqmap, mask))
        with self.assertRaises(AssertionError):
            qpm.select_sort()

    def test_sorted_qpm_feeds_twotime_layout(self):
        dqmap, sqmap, mask = make_maps()
        qpm = XpcsQPartitionMap(self.write_qmap(dqmap, sqmap, mask), flag_sort=True)
        idx = qpm.info_np["mask_idx_1d"]
        for dq, slc in zip(qpm.qinfo["dq_idx"], qpm.qinfo["dq_slc"]):
            self.assertTrue(np.all(dqmap.ravel()[idx[slc]] == dq))
        for sq, slc in zip(qpm.qinfo["sq_idx"], qpm.qinfo["sq_slc"]):
            self.assertTrue(np.all(sqmap.ravel()[idx[slc]] == sq))


if __name__ == "__main__":
    unittest.main()
