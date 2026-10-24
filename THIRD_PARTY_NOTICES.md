# Third-party attribution

The project MIT License covers original ParaPC-FM contributions. It does not
relicense third-party code, dependencies or model weights.

## ParaDiGMS

The Flow sliding-window sampler follows the ParaDiGMS framework from
[AndyShih12/paradigms](https://github.com/AndyShih12/paradigms).

Source version: `d44faf9ae6757d54148711f58721d4b399ac28a6`:
[`stablediffusion_paradigms.py`](https://github.com/AndyShih12/paradigms/blob/d44faf9ae6757d54148711f58721d4b399ac28a6/paradigms/stablediffusion_paradigms.py)
and
[`stablediffusion_paradigms_mp.py`](https://github.com/AndyShih12/paradigms/blob/d44faf9ae6757d54148711f58721d4b399ac28a6/paradigms/stablediffusion_paradigms_mp.py).

Copyright (c) 2023 Andy Shih. Licensed under the MIT License, reproduced below.

Paper: Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh and Nima Anari,
*Parallel Sampling of Diffusion Models*, NeurIPS 2023.

ParaPC-FM adds Flow Matching updates, step-size-based error normalization,
predictor-corrector updates and truncated initialization. The distributed
runtime uses PyTorch process groups and collectives.

### ParaDiGMS license

```text
MIT License

Copyright (c) 2023 Andy Shih

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ParaTAA

The SD1.5 triangular Anderson sampler and DDIM grid coefficient construction
build on [ParaTAA-Diffusion](https://github.com/TZW1998/ParaTAA-Diffusion),
with ParaPC-FM initialization extensions.

Upstream source: [`parallel_sd.py`](https://github.com/TZW1998/ParaTAA-Diffusion/blob/08bcf3fdd264cc612215d88deb6e3fab5fe26a3c/parallel_sd.py),
commit `08bcf3fdd264cc612215d88deb6e3fab5fe26a3c`.

Upstream credited authors: Zhiwei Tang, Jiasheng Tang, Hao Luo, Fan Wang and
Tsung-Hui Chang. Paper: *Accelerating Parallel Sampling of Diffusion Models*,
ICML 2024. https://openreview.net/forum?id=CjVWen8aJL

Upstream-derived portions of `parapc_fm/models/sd15/parataa_sampler.py` and
`build_ddim_grid` in `parapc_fm/prediction/parataa_predictor.py` are excluded
from the project MIT grant. No upstream license was declared at the referenced
commit; redistribution permission for those portions remains unresolved.

## Diffusers and model weights

The adapters call Hugging Face Diffusers (Apache-2.0); they use its public model
and pipeline objects and are not copies of complete Diffusers pipeline files.
Keep dependency attribution when distributing dependencies.

FLUX and Stable Diffusion weights are not bundled. Each model's own access and
license conditions continue to apply. A software license does not grant rights
to redistribute model weights or datasets.

## Original project

Flow update/initialization and window conventions are derived from this
project's research implementation.
