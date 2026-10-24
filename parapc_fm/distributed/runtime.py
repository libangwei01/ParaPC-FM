"""One model replica per process; distribute timesteps of the same image."""

import os
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, device=None):
        self.owns_group = False
        if device is None:
            device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
            self.owns_group = True
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def is_main(self):
        return self.rank == 0

    def broadcast(self, value):
        if self.world_size > 1:
            dist.broadcast(value, src=0)
        return value

    def evaluate(self, indices, template, function):
        """Evaluate an ordered timestep list, including uneven and empty shards.

        All ranks allocate the full small velocity buffer and SUM disjoint
        contributions. No padded fake samples or duplicate model calls are used.
        """
        output = torch.zeros((len(indices), *template.shape), device=template.device,
                             dtype=template.dtype)
        for slot in range(self.rank, len(indices), self.world_size):
            output[slot] = function(indices[slot])
        if self.world_size > 1:
            dist.all_reduce(output)
        return output

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.world_size > 1:
            dist.barrier()

    def maximum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
