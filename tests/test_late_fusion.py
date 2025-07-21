import unittest
import torch
from model import LateFusionViT

class TestLateFusionViT(unittest.TestCase):
    def test_feature_concat_fusion(self):
        model = LateFusionViT(fusion_type='concat', fusion_layer='feature')
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        out = model(rgb, thermal)
        self.assertEqual(out['logits'].shape, (2, 7))

    def test_prediction_attention_fusion(self):
        model = LateFusionViT(fusion_type='attention', fusion_layer='prediction')
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        out = model(rgb, thermal)
        self.assertEqual(out['logits'].shape, (2, 7))

    def test_forward_with_labels_returns_loss(self):
        model = LateFusionViT()
        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        thermal = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 7, (2,))
        out = model(rgb, thermal, labels)
        self.assertIsNotNone(out['loss'])

    def test_freeze_unfreeze(self):
        model = LateFusionViT(freeze_backbone=True)
        self.assertTrue(all(not p.requires_grad for p in model.rgb_vit.parameters()))
        model.unfreeze_backbone()
        self.assertTrue(any(p.requires_grad for p in model.rgb_vit.parameters()))

if __name__ == "__main__":
    unittest.main()
