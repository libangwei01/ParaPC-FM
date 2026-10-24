import unittest
import torch
from parapc_fm.distributed import Runtime
from parapc_fm.sampling import sample_flow


class WindowTests(unittest.TestCase):
    def test_original_nips_regression_all_nine_combinations(self):
        # Frozen outputs from the original nips loop and predict_point.py,
        # x'=.1*x+.3, 7 uniform decreasing steps, window=3, tolerance=.01.
        expected = [
            -.28742848812994587, -.287429237817576, -.2868805758017493,
            -.28571962203248646, -.28572730633069554, -.28547049146189085,
            -.28544218294460644, -.28572253748438153, -.2854669213348604,
        ]
        for p in range(3):
            for q in range(3):
                final, iterations, _ = sample_flow(torch.zeros(1, 2, dtype=torch.float64),
                    torch.linspace(1, 0, 8, dtype=torch.float64), lambda x, i: .1*x+.3,
                    Runtime("cpu"), update_order=p, init_order=q, window_size=3, tolerance=.01)
                torch.testing.assert_close(final, torch.full_like(final, expected[3*p+q]), rtol=1e-12, atol=1e-12)
                self.assertEqual(iterations, 4)

    def test_all_orders_constant_velocity_and_batch(self):
        for p in range(3):
            for q in range(3):
                for window in (1, 3, 8):
                    initial = torch.zeros(2, 3, dtype=torch.float64)
                    times = torch.linspace(1, 0, 8, dtype=torch.float64)
                    final, iterations, evaluations = sample_flow(initial, times,
                        lambda state, i: torch.ones_like(state), Runtime("cpu"),
                        update_order=p, init_order=q, window_size=window, tolerance=0)
                    torch.testing.assert_close(final, -torch.ones_like(initial))
                    self.assertLessEqual(iterations, 7)
                    self.assertGreaterEqual(evaluations, iterations)

    def test_bad_orders_and_time_grid(self):
        initial = torch.zeros(1, 2)
        for kwargs in ({"update_order": 3}, {"init_order": -1}, {"window_size": 0}, {"tolerance": -1}):
            with self.assertRaises(ValueError):
                sample_flow(initial, torch.tensor([1., 0.]), lambda x, i: x, Runtime("cpu"), **kwargs)
        with self.assertRaises(ValueError):
            sample_flow(initial, torch.tensor([1., 1., 0.]), lambda x, i: x, Runtime("cpu"))


if __name__ == "__main__":
    unittest.main()
