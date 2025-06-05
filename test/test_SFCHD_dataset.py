import os
import sys

# Add the modules directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset')))

import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from dataset import SFCHD

class TestSFCHD(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the dataset
        self.test_dir = tempfile.mkdtemp()
        self.image_folder = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_folder, exist_ok=True)

        # Create dummy train.txt
        self.train_txt = os.path.join(self.test_dir, "train.txt")
        with open(self.train_txt, "w") as f:
            f.write("sample1.txt\n")
            f.write("sample2.txt\n")

        # Create dummy annotation files
        for i in range(1, 3):
            ann_path = os.path.join(self.test_dir, f"sample{i}.txt")
            with open(ann_path, "w") as f:
                f.write(f"{i-1} 10 20 30 40\n")

        # Create dummy images
        for i in range(1, 3):
            img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
            img.save(os.path.join(self.image_folder, f"sample{i}.jpg"))

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_len(self):
        dataset = SFCHD(self.test_dir, self.image_folder, train=True)
        self.assertEqual(len(dataset), 2)

    def test_getitem(self):
        dataset = SFCHD(self.test_dir, self.image_folder, train=True)
        image, target = dataset[0]
        from PIL.Image import Image as PILImage
        self.assertTrue(isinstance(image, PILImage))
        self.assertIn('class_ids', target)
        self.assertIn('bboxes', target)
        self.assertEqual(len(target['class_ids']), 1)
        self.assertEqual(len(target['bboxes']), 1)

if __name__ == '__main__':
    unittest.main()