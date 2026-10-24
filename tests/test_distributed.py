"""Real Gloo collectives on two CPU processes; no model weights required."""
import tempfile
import unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parapc_fm.distributed import Runtime
from parapc_fm.sampling import sample_flow


def worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        runtime = Runtime("cpu")
        for length in (1, 3, 5):
            values = runtime.evaluate(list(range(length)), torch.zeros(1, 2),
                                      lambda i: torch.full((1, 2), float(i + 1)))
            torch.testing.assert_close(values[:, 0, 0], torch.arange(1, length + 1).float())
        results = []
        for p in range(3):
            for q in range(3):
                result = sample_flow(torch.zeros(1, 2, dtype=torch.float64),
                    torch.linspace(1, 0, 8, dtype=torch.float64),
                    lambda x, i: .1*x + .3, runtime,
                    update_order=p, init_order=q, window_size=3, tolerance=.01)
                results.append(result)
        if rank == 0:
            torch.save(results, output)
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_two_ranks_equal_serial_all_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "result.pt")
            mp.spawn(worker, args=(str(Path(directory) / "rendezvous"), output), nprocs=2, join=True)
            results = torch.load(output, weights_only=True)
            for (p, q), distributed in zip(((p, q) for p in range(3) for q in range(3)), results):
                serial = sample_flow(torch.zeros(1, 2, dtype=torch.float64),
                    torch.linspace(1, 0, 8, dtype=torch.float64),
                    lambda x, i: .1*x + .3, Runtime("cpu"),
                    update_order=p, init_order=q, window_size=3, tolerance=.01)
                torch.testing.assert_close(distributed[0], serial[0])
                self.assertEqual(distributed[1:], serial[1:])


if __name__ == "__main__":
    unittest.main()
