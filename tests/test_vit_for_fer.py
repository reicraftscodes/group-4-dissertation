import unittest
import torch
from model import ViTForFER

class TestViTForFER(unittest.TestCase):
    def setUp(self):
        self.model = ViTForFER()
        self.model.eval()

    def test_forward_logits_shape(self):
        x = torch.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 7))

    def test_forward_with_labels_returns_loss(self):
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 7, (2,))
        out = self.model(x, labels=y)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.dim(), 0)

    def test_feature_extraction_shape(self):
        x = torch.randn(2, 3, 224, 224)
        features = self.model.get_features(x)
        self.assertEqual(features.shape[0], 2)

    def test_attention_output(self):
        x = torch.randn(2, 3, 224, 224)
        attn = self.model.get_attention_weights(x)
        self.assertTrue(isinstance(attn, torch.Tensor))

    def test_freeze_unfreeze_backbone(self):
        self.model._freeze_backbone()
        self.assertTrue(all(not p.requires_grad for p in self.model.vit.vit.parameters()))
        self.model.unfreeze_backbone()
        self.assertTrue(any(p.requires_grad for p in self.model.vit.vit.parameters()))

    def test_model_info_keys(self):
        info = self.model.get_model_info()
        expected_keys = ['model_name', 'num_classes', 'total_parameters', 'trainable_parameters']
        for key in expected_keys:
            self.assertIn(key, info)

if __name__ == "__main__":
    unittest.main()
