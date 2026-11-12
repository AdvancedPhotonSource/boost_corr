#!/usr/bin/env python

"""Tests for twotime correlation."""


import unittest
from boost_corr.correlator import TwotimeCorrelator
import torch
import numpy as np


def make_qinfo():
    qinfo = {}
    qinfo['dq_idx'] = np.arange(1, 5)    # 1, 2, 3, 4
    qinfo['dq_slc'] = [slice(x * 5 - 5, x * 5) for x in qinfo['dq_idx']]
    qinfo['sq_idx'] = np.arange(1, 21)   # 1, 2, ... ,20
    qinfo['sq_slc'] = [slice(x - 1, x) for x in qinfo['sq_idx']]
    return qinfo


def run_and_get_result(frame_num=512, device='cpu', smooth_method='sqmap'):
    det_size = (4, 5)
    qinfo = make_qinfo()

    num_elm = det_size[0] * det_size[1]
    tt = TwotimeCorrelator(qinfo=qinfo, det_size=det_size,
                           frame_num=frame_num, device=device)
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device).reshape(-1, num_elm)
        tt.process(x)
    tt.post_process(smooth_method=smooth_method)
    result_saxs, twotime_generator = tt.get_results()
    # the c2 maps are computed while the generator is consumed
    result_twotime = list(twotime_generator)
    return result_saxs, result_twotime


class TestBoost_corr(unittest.TestCase):
    """Tests for `boost_corr` package."""

    def check_uniform_frames(self, device, smooth_method):
        frame_num = 1024
        saxs, twotime = run_and_get_result(frame_num=frame_num, device=device,
                                           smooth_method=smooth_method)

        saxs2d = saxs['saxs_2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        saxs2d_par = saxs['saxs_2d_par']
        saxs2d_par_real = torch.ones_like(saxs2d_par)
        self.assertEqual(torch.allclose(saxs2d_par, saxs2d_par_real), True)

        intt = saxs['intensity_vs_time'][1]
        self.assertEqual(torch.allclose(intt, torch.ones_like(intt)), True)

        # one c2 map per dynamic q bin, then the g2 summary
        c2_keys = []
        for d in twotime[:-1]:
            for k, v in d.items():
                c2_keys.append(k)
                self.assertEqual(v.shape, (frame_num, frame_num))
                v_real = np.zeros_like(v)
                v_real[np.triu_indices(v_real.shape[0])] = 1
                self.assertEqual(np.allclose(v, v_real), True)
        self.assertEqual(
            c2_keys, [f'correlation_map/c2_{n:05d}' for n in range(1, 5)])

        summary = twotime[-1]
        self.assertEqual(np.all(summary['processed_bins'] == np.arange(1, 5)),
                         True)
        c2_g2 = summary['c2_g2']
        self.assertEqual(c2_g2.shape, (frame_num, 4))
        self.assertEqual(np.allclose(c2_g2, np.ones_like(c2_g2)), True)

    def test_000_cpu_sqmap(self):
        self.check_uniform_frames('cpu', 'sqmap')

    def test_001_cpu_dqmap(self):
        self.check_uniform_frames('cpu', 'dqmap')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_002_gpu(self):
        self.check_uniform_frames('cuda:0', 'sqmap')

    def test_003_describe(self):
        # run_jobs calls describe() on the correlator when verbose is set
        tt = TwotimeCorrelator(qinfo=make_qinfo(), det_size=(4, 5),
                               frame_num=16, device='cpu')
        with self.assertLogs('boost_corr.correlator.twotime', 'INFO') as cm:
            tt.describe()
        self.assertIn('twotime: 4 dynamic q bins', cm.output[0])

    def test_004_all_correlators_describe(self):
        from boost_corr.correlator import EventCorrelator, MultitauCorrelator
        for cls in (MultitauCorrelator, TwotimeCorrelator, EventCorrelator):
            self.assertTrue(callable(getattr(cls, 'describe', None)), cls)


if __name__ == '__main__':
    unittest.main()
