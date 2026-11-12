#!/usr/bin/env python

"""Tests for metadata file discovery."""


import unittest
from boost_corr.xpcs_aps_8idi.exceptions import MetadataError
from boost_corr.xpcs_aps_8idi.xpcs_result import get_metadata, is_metadata
import os
import tempfile
import h5py


class TestBoost_corr(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_dir_fname = self.temp_dir.name
        # get_metadata only looks at files named *_metadata.hdf
        flist = [os.path.join(temp_dir_fname, x + '_metadata.hdf') for x in [
            'false', 'true']]

        with h5py.File(flist[0], 'w') as f:
            f['/exchange/data'] = 1.0
        with h5py.File(flist[1], 'w') as f:
            f['/entry/schema_version'] = 2.0
        self.flist = flist

    def test_000_is_meta(self):
        self.assertEqual(is_metadata(self.flist[0]), (False, None))
        self.assertEqual(is_metadata(self.flist[1]), (True, 'nexus'))
        self.assertEqual(is_metadata(''), (False, None))

    def test_001_get_meta(self):
        meta_fname, meta_type = get_metadata(self.temp_dir.name)
        self.assertEqual(meta_fname, self.flist[1])
        self.assertEqual(meta_type, 'nexus')

    def test_002_not_found(self):
        os.remove(self.flist[1])
        self.assertRaises(MetadataError,
                          lambda: get_metadata(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
