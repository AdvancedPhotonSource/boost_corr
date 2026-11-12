#!/usr/bin/env python

"""Tests for multitau correlation."""


import unittest
from boost_corr.correlator import MultitauCorrelator
from boost_corr.correlator.utils import gen_tau_bin
import torch
import numpy as np


def run_and_get_result(det_size=(5, 7), frame_num=1024, device='cpu'):
    num_elm = det_size[0] * det_size[1]
    mc = MultitauCorrelator(det_size, frame_num, device=device,
                            max_count=4096)
    # use high max_count to make sure all levels are float32
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device)
        mc.process(x.reshape(-1, num_elm))
    mc.post_process()
    return mc.get_results()


class TestBoost_corr(unittest.TestCase):
    """Tests for `boost_corr` package."""

    def check_uniform_frames(self, device):
        frame_num = 1024
        scattering, multitau = run_and_get_result(frame_num=frame_num,
                                                  device=device)
        intt = scattering['intensity_vs_time'][1]
        intt_real = torch.ones_like(intt)
        self.assertEqual(torch.allclose(intt, intt_real), True)

        saxs2d = scattering['saxs_2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        tau = multitau['tau']
        tau_bin = gen_tau_bin(frame_num)
        self.assertEqual(np.all(tau == tau_bin[0]), True)

        # G2, IP and IF per tau and pixel; g2 = G2 / (IP * IF)
        G2 = multitau['G2']
        self.assertEqual(tuple(G2.shape), (tau_bin.shape[1], 3, 5 * 7))
        self.assertEqual(torch.allclose(G2, torch.ones_like(G2)), True)
        g2 = G2[:, 0] / (G2[:, 1] * G2[:, 2])
        self.assertEqual(torch.allclose(g2, torch.ones_like(g2)), True)

    def test_000_cpu(self):
        self.check_uniform_frames('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_001_gpu(self):
        self.check_uniform_frames('cuda:0')


if __name__ == '__main__':
    unittest.main()
