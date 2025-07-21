import unittest
from model import create_multimodal_vit_model

class TestModelFactory(unittest.TestCase):
    def test_create_rgb_model(self):
        model = create_multimodal_vit_model(mode='rgb')
        self.assertIsNotNone(model)

    def test_create_combined_early(self):
        model = create_multimodal_vit_model(mode='combined', fusion_strategy='early', fusion_type='add')
        self.assertIsNotNone(model)

    def test_create_combined_late(self):
        model = create_multimodal_vit_model(mode='combined', fusion_strategy='late', fusion_type='attention', fusion_layer='prediction')
        self.assertIsNotNone(model)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            create_multimodal_vit_model(mode='invalid')

if __name__ == "__main__":
    unittest.main()
