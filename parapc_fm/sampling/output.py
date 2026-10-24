from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingOutput:
    images: Any
    iterations: int
    model_evaluations: int
    timings: dict = field(default_factory=dict)
