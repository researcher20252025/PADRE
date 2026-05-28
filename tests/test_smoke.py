import unittest

import torch

from padre.config import ExperimentConfig
from padre.data import segment_indices
from padre.models import ITransformerForecaster, PatchTSTForecaster, TCNForecaster


class SmokeTests(unittest.TestCase):
    def test_config_defaults(self) -> None:
        cfg = ExperimentConfig(data_path="ETTh2.csv", horizon=24)
        self.assertEqual(cfg.backbone, "tcn")
        self.assertEqual(cfg.lookback, 60)

    def test_segment_indices_are_chronological(self) -> None:
        starts = segment_indices(lookback=4, horizon=2, start_target=4, end_target=12)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4, 5, 6])

    def test_tcn_forward_shape(self) -> None:
        model = TCNForecaster(
            lookback=16,
            input_dim=5,
            target_dim=3,
            horizon=4,
            hidden=8,
            levels=1,
            head_mode="summary",
        )
        y = model(torch.randn(2, 16, 5))
        self.assertEqual(tuple(y.shape), (2, 4, 3))

    def test_patchtst_forward_shape(self) -> None:
        model = PatchTSTForecaster(
            lookback=32,
            horizon=4,
            input_dim=5,
            target_dim=3,
            d_model=8,
            n_heads=2,
            layers=1,
            dim_ff=16,
            patch_len=8,
            stride=4,
            dropout=0.0,
            revin=False,
            individual_heads=False,
        )
        y = model(torch.randn(2, 32, 5))
        self.assertEqual(tuple(y.shape), (2, 4, 3))

    def test_itransformer_forward_shape(self) -> None:
        model = ITransformerForecaster(
            lookback=32,
            horizon=4,
            input_dim=5,
            target_dim=3,
            d_model=8,
            n_heads=2,
            layers=1,
            dim_ff=16,
            dropout=0.0,
            revin=False,
            token_mode="all",
            linear_skip=False,
        )
        y = model(torch.randn(2, 32, 5))
        self.assertEqual(tuple(y.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
