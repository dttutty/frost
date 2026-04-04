"""Tests for RingBuf shared-memory circular buffer."""

import pytest
import torch

from frost.shm.ring_buf import RingBuf, cal_max_buf_size


class TestCalMaxBufSize:
    def test_no_fanout(self):
        n_src, n_neg = 100, 1
        size = cal_max_buf_size(n_src, fanout=None, n_neg=n_neg)
        expected = n_src * (2 + n_neg)
        assert size == expected

    def test_single_fanout(self):
        n_src, n_neg, fanout = 100, 1, [10]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_multi_hop_fanout(self):
        n_src, n_neg, fanout = 50, 2, [10, 5]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_1hop = n_root * fanout[0]
        n_2hop = n_1hop * fanout[1]
        assert size == n_root + n_1hop + n_2hop

    def test_sample_4_neg_false(self):
        n_src, n_neg, fanout = 100, 1, [10]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg, sample_4_neg=False)
        n_root = n_src * (2 + n_neg)
        expected = n_neg + n_root + n_root * fanout[0]
        assert size == expected

    def test_zero_neg(self):
        n_src, n_neg, fanout = 100, 0, [5]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_large_fanout(self):
        n_src, n_neg, fanout = 10, 1, [100]
        size = cal_max_buf_size(n_src, fanout=fanout, n_neg=n_neg)
        n_root = n_src * (2 + n_neg)
        n_nbr = n_root * fanout[0]
        assert size == n_root + n_nbr

    def test_empty_fanout_list(self):
        n_src, n_neg = 100, 1
        size = cal_max_buf_size(n_src, fanout=[], n_neg=n_neg)
        expected = n_src * (2 + n_neg)
        assert size == expected
