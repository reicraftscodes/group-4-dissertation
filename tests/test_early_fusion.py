import unittest
import torch
from model import EarlyFusionViT

class TestEarlyFusionViT(unittest.TestCase):
    def setUp(self):
        self.model = EarlyFusionViT()
        self.model.eval()

    def test_forward_concat_fusion(self):
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        out = self.model(rgb, thermal)
        self.assertEqual(out['logits'].shape, (2, 7))

    def test_forward_with_labels_returns_loss(self):
        model = EarlyFusionViT()
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 7, (2,))
        out = model(rgb, thermal, labels=labels)
        self.assertIsNotNone(out['loss'])

    def test_fusion_add_mode(self):
        model = EarlyFusionViT(fusion_type='add')
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        out = model(rgb, thermal)
        self.assertEqual(out['logits'].shape, (2, 7))

if __name__ == "__main__":
    unittest.main()
