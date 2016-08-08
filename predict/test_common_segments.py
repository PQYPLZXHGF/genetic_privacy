#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

import numpy as np

# import pyximport; pyximport.install()
from common_segments import common_homolog_segments, _consolidate_sequence

uint32 = np.uint32

class TestCommonHomologSegments(unittest.TestCase):
    def test_single_same_segment(self):
        a = MagicMock()
        a.starts = np.array([0], dtype = uint32)
        a.founder = np.array([0], dtype = uint32)
        a.end = 10
        shared = common_homolog_segments(a, a)
        self.assertEqual(shared, [(0, 10)])

    def test_single_different_segment(self):
        a = MagicMock()
        a.starts = np.array([0], dtype = uint32)
        a.founder = np.array([0], dtype = uint32)
        a.end = 10
        b = MagicMock()
        b.starts = np.array([0], dtype = uint32)
        b.founder = np.array([1], dtype = uint32)
        b.end = 10
        shared = common_homolog_segments(a, b)
        self.assertEqual(shared, [])

    def test_multiple_same_segment(self):
        a = MagicMock()
        a.starts = np.array([0], dtype = uint32)
        a.founder = np.array([0], dtype = uint32)
        a.end = 10
        b = MagicMock()
        b.starts = np.array([0, 5], dtype = uint32)
        b.founder = np.array([0, 0], dtype = uint32)
        b.end = 10
        shared = common_homolog_segments(a, b)
        self.assertEqual(shared, [(0, 10)])

    def test_two_same_segments(self):
        a = MagicMock()
        a.starts = np.array([0, 5], dtype = uint32)
        a.founder = np.array([1, 2], dtype = uint32)
        a.end = 10
        shared = common_homolog_segments(a, a)
        self.assertEqual(shared, [(0, 10)])

    def test_two_segments_different_boundary(self):
        a = MagicMock()
        a.starts = np.array([0, 5], dtype = uint32)
        a.founder = np.array([1, 1], dtype = uint32)
        a.end = 10
        b = MagicMock()
        b.starts = np.array([0, 6], dtype = uint32)
        b.founder = np.array([1, 1], dtype = uint32)
        b.end = 10
        shared = common_homolog_segments(a, b)
        self.assertEqual(shared, [(0, 10)])

    def test_single_element_vs_many_match_in_back(self):
        a = MagicMock()
        a.starts = np.array([0], dtype = uint32)
        a.founder = np.array([0], dtype = uint32)
        a.end = 10
        b = MagicMock()
        b.starts = np.array([0, 2, 4, 8], dtype = uint32)
        b.founder = np.array([1, 2, 3, 0], dtype = uint32)
        b.end = 10
        shared = common_homolog_segments(a, b)
        self.assertEqual(shared, [(8, 10)])

    def test_single_element_vs_many_match_in_front(self):
        a = MagicMock()
        a.starts = np.array([0], dtype = uint32)
        a.founder = np.array([0], dtype = uint32)
        a.end = 10
        b = MagicMock()
        b.starts = np.array([0, 2, 4, 8], dtype = uint32)
        b.founder = np.array([0, 2, 3, 4], dtype = uint32)
        b.end = 10
        shared = common_homolog_segments(a, b)
        self.assertEqual(shared, [(0, 2)])

class TestConsolidateSequence(unittest.TestCase):
    def test_two_elements_merge(self):
        seq = [(0, 5), (5, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 10)])

    def test_two_elements_disjoint(self):
        seq = [(0, 5), (6, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, seq)

    def test_first_two_merge(self):
        seq = [(0, 4), (4, 8), (9, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 8), (9, 10)])

    def test_last_two_merge(self):
        seq = [(0, 4), (5, 8), (8, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 4), (5, 10)])
        
    def test_middle_two_merge(self):
        seq = [(0, 3), (4, 6), (6, 8), (9, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 3), (4, 8), (9, 10)])

    def test_many_elements(self):
        seq = [(0, 2), (2, 4), (4, 8), (8, 10)]
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 10)])

    def test_many_emements_single_point(self):
        seq = list(zip(range(0, 10), range(1, 11)))
        con = _consolidate_sequence(seq)
        self.assertEqual(con, [(0, 10)])

if __name__ == '__main__':
    unittest.main()
