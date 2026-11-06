import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

from app.pipeline.dataset_loader.storage_dataset_loader import StorageDatasetLoader


class DatasetLoaderTests(unittest.TestCase):
    def test_local_and_blob_parse_identical_semicolon_csv(self):
        contents = 'fecha;producto;cantidad\n2026-01-01;PARACETAMOL;3\n'.encode()
        loader = StorageDatasetLoader()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sales.csv'
            path.write_bytes(contents)
            local = loader.load(str(path)).dataframe
            with patch('app.pipeline.dataset_loader.storage_dataset_loader.urlopen', return_value=io.BytesIO(contents)):
                blob = loader.load('https://example.invalid/sales.csv?sig=secret').dataframe
        self.assertTrue(local.equals(blob))
        self.assertEqual(list(blob.columns), ['fecha', 'producto', 'cantidad', '_source_file'])

    def test_blob_error_does_not_expose_sas(self):
        with patch('app.pipeline.dataset_loader.storage_dataset_loader.urlopen', side_effect=OSError('sig=secret')):
            with self.assertRaises(FileNotFoundError) as context:
                StorageDatasetLoader().load('https://example.invalid/sales.csv?sig=secret')
        self.assertNotIn('secret', str(context.exception))


if __name__ == '__main__':
    unittest.main()
