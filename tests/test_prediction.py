import unittest
from types import SimpleNamespace
import torch
from parapc_fm.prediction import correction_weights, initialize_future_states, initialize_ddim_states
from parapc_fm.prediction.paradigms_predictor import _increment


class PredictionTests(unittest.TestCase):
    def test_corrector_polynomial_exactness_all_p(self):
        # Nonuniform decreasing grid, x'=t^p. Degree p is integrated exactly.
        times = torch.tensor([1., .83, .51, .17, 0.], dtype=torch.float64)
        for p in range(3):
            velocities = times.pow(p).reshape(-1, 1, 1)
            actual = _increment(velocities, times, 2, p, corrector=True)
            expected = (times[3]**(p+1) - times[2]**(p+1)) / (p+1)
            torch.testing.assert_close(actual.squeeze(), expected)

    def test_prediction_q_zero_one_two(self):
        for direction in (1, -1):
            times = direction * torch.tensor([0., .1, .3, .6, 1.], dtype=torch.float64)
            for q in range(3):
                states = torch.zeros(5, 1, 1, dtype=torch.float64)
                velocities = torch.ones_like(states) if q < 2 else times.reshape(-1, 1, 1)
                if q == 0:
                    states[2] = 4
                initialize_future_states(states, velocities, times, 2, 4, q)
                if q == 0:
                    expected = torch.full((3,), 4., dtype=torch.float64)
                elif q == 1:
                    expected = times[2:] - times[1]
                else:
                    expected = (times[2:]**2 - times[1]**2) / 2
                torch.testing.assert_close(states[2:, 0, 0], expected)

    def test_q_two_without_history_falls_back_to_linear(self):
        times = torch.tensor([1., .5, 0.], dtype=torch.float64)
        states = torch.ones(3, 1, 1, dtype=torch.float64)
        velocities = torch.full_like(states, 2.)
        initialize_future_states(states, velocities, times, 1, 2, 2)
        torch.testing.assert_close(states[:, 0, 0], torch.tensor([1., 0., -1.], dtype=torch.float64))

    def test_ddim_all_q_and_boundary(self):
        for q in range(3):
            for reference in (2, 4):
                times = torch.tensor([1., 2., 4., 7., 11.], dtype=torch.float64)
                states = torch.zeros(6, 1, 1, 1, dtype=torch.float64)
                states[:, 0, 0, 0] = torch.arange(6, dtype=torch.float64)**2
                original = states.clone()
                sampler = SimpleNamespace(uninitialized_index=reference-1, finished_index=5,
                    window_size=8, _samples=states, scheduler_timesteps=times, new_image=False)
                initialize_ddim_states(sampler, q)
                self.assertTrue(torch.isfinite(states).all())
                if q == 0 or reference == 4:
                    torch.testing.assert_close(states[:reference], original[reference].expand(reference, 1, 1, 1))
                elif q == 1:
                    velocity = (original[reference] - original[reference+1]) / (times[reference] - times[reference+1])
                    for j in range(reference):
                        expected = original[reference] + (times[j]-times[reference])*velocity
                        torch.testing.assert_close(states[j], expected)
                else:
                    velocity = (original[2]-original[3]) / (times[2]-times[3])
                    previous = (original[3]-original[4]) / (times[3]-times[4])
                    for j in range(reference):
                        h = times[j]-times[2]
                        expected = original[2] + h*velocity + h*h/(2*(times[3]-times[2]))*(previous-velocity)
                        torch.testing.assert_close(states[j], expected)

    def test_invalid_step(self):
        with self.assertRaises(ValueError):
            correction_weights(torch.tensor([1.]), 0.)


if __name__ == "__main__":
    unittest.main()
