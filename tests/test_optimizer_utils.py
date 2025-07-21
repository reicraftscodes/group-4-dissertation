import unittest
from model import create_multimodal_vit_model, get_optimizer_and_scheduler

class TestOptimizerUtils(unittest.TestCase):
    def test_optimizer_scheduler(self):
        model = create_multimodal_vit_model(mode='rgb')
        optimizer, scheduler = get_optimizer_and_scheduler(model)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

if __name__ == "__main__":
    unittest.main()
