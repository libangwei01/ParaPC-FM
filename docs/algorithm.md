# Orders and numerical conventions

## Flow update p

Write h=t[i+1]-t[i], with signed h (normally negative for the supplied models).
Every network velocity in one window is evaluated on the old iterate before
any states are overwritten. New states are accumulated from the updated anchor.

- p=0: h*v[i].
- p=1: h*v[i] + h/2*(v[i+1]-v[i]); this is the paper's Heun-style
  fixed-point corrector, not an additional sequential Heun model evaluation.
- p=2: h*v[i] + a1*(v[i-1]-v[i]) + a2*(v[i+1]-v[i]).
  Solve the two-moment system with offsets (t[i-1]-t[i])/h and 1.

At the first step, unavailable backward terms are omitted. The last step of a
window has no v[end]; use only available backward terms (Euler for p=0/1).
This is a predictor boundary, not a full corrector. Coefficient systems are
solved in float64 and cast to the velocity dtype for application.

Public `update_order` accepts paper p directly.

## Initialization q

For reference ell, target j, H=t[j]-t[ell]:

- q=0: constant state copy.
- q=1: x[ell] + H*v[ell].
- q=2: x[ell] + H*v[ell] + H²/(2*(t[ell-1]-t[ell]))*(v[ell-1]-v[ell]).

When history is unavailable, q=2 reduces to q=1. In the Flow implementation,
the last evaluated velocity is at ell=old_end-1. q>0 refreshes the interval
`[old_end, new_end]`, including the old endpoint; q=0 copies x[old_end]. This
overlap also occurs on the last window. The initial window starts with constant states for every q,
because no velocities have been evaluated yet.

## Sliding-window update

The reference anchor at begin is corrected from the preceding state when p>0
and begin>0. Accept through the first failed threshold index (always at least
one step). Rebuild the unconverged tail with the predictor for p=2, then
initialize newly exposed states. This is not a generic off-the-shelf UniPC
scheduler; do not replace these operations with scheduler.step calls.

The convergence criterion is mean squared latent difference / h² <= tol².
The public adapters support one image; the numerical engine also requires all
images in a supplied batch to pass. Error is normalized independently per image.

## SD1.5 / ParaTAA

ParaTAA uses a reversed DDIM timestep grid and epsilon predictions. Its
initialization slopes are finite differences of stored latent states; they
are not Flow model velocities. The p formulas above do not apply to ParaTAA's
`taa_order`. DDIM grid coefficients and initialization are in `prediction/`;
the inherited triangular Anderson update remains in the sampler.

Initialization uses signed time displacements for both linear and historical
terms. Missing history falls back first to a constant state, then to a linear
prediction. CFG embeddings are ordered `[negative, positive]`, and network
outputs are unpacked as `[unconditional, conditional]`. Sampler history is
reset per image. Reported iteration counts include correction rounds only.
