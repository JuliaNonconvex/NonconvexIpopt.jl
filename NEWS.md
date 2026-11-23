# NEWS

## 0.9.2

- Docs: the logo now appears only in the sidebar, not in the page body.
## 0.9.1

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.9.0

- New `lagrangian_hessian` option (and `IpoptWorkspace`/`optimize` keyword): a caller-provided `(x, obj_factor, lambda) -> H` callback replaces the AD path in `eval_h`, so solver frameworks can supply exact second derivatives assembled from MathOptInterface evaluators without nested AD. When provided, the Hessian sparsity pattern is the dense lower triangle (active-set safe).
- Option fields are read with their documented defaults, so options built through the raw `IpoptOptions(nt::NamedTuple)` constructor (a partial field set) are valid.
## 0.8.6

- Dual leaks from user functions that differentiate internally now hard-error
  with a descriptive `ArgumentError` (naming the offending quantity) instead
  of being silently stripped: a `ForwardDiff.Dual` reaching a Float64
  callback is a nested-AD artifact in the user's function, not something the
  wrapper should paper over.

## 0.8.5

- `Internal_Error` is mapped from -199 (the -999 branch was dead) and `Maximum_WallTime_Exceeded` (-5) is mapped.\n- A runtime Hessian sparsity mismatch in `eval_h` falls back to the dense Hessian instead of crashing.
## 0.8.4

- Added docstrings for `IpoptAlg` and `IpoptOptions` (REPL help and API reference).


## 0.8.3

- Fixed the default options, the no-`x0` Workspace path and the callback
  Solution aliasing.

## 0.8.2

- Skipped the dense AD Jacobian on the JuMP-converted fast path.

## 0.8.1

- Stripped the nested-AD duals from the Float64 buffers.

## 0.8.0

- NonconvexCore 2 with strict solver capability declarations, sparse AD via
  the `sparse` option, and symbolic differentiation via the `symbolic`
  option.

## 0.7.1

- `optimize` now accepts the generic `callback` keyword. The callback runs
  once per Ipopt iteration through Ipopt's intermediate callback and receives
  a `Solution` carrying the current iterate (from
  `Ipopt.GetIpoptCurrentIterate`), the objective value, the constraint values
  and multipliers, and the primal/dual infeasibility in the convergence
  state.
