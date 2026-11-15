# Changelog

## 0.8.3

- Fixed the default options, the no-`x0` Workspace path and the callback
  Solution aliasing; covered the JuMP-converted evaluator fast path and the
  option branches; scheduled formatter workflow.

## 0.8.2

- Skipped the dense AD Jacobian on the JuMP-converted fast path.

## 0.8.1

- Stripped the nested-AD duals from the Float64 buffers; codecov OIDC/v5
  uploads; logo and README polish.

## 0.8.0 (2026-11-10)

- Support NonconvexCore 2: `NonconvexCore = "2"` is now the compat lower bound
- Declare strict solver capabilities (`linear_eq`, `nonlinear_eq`,
  `linear_ineq`, `nonlinear_ineq`, `linear_sd`, `nonlinear_sd`, `box`,
  `continuous`, `binary`, `integer`, `infinite_bounds`)
- Sparse AD support via the `sparse` option and symbolic differentiation via
  the `symbolic` option
- Test suite reorganized with TestItems and a separate test environment;
  Aqua and JET quality checks; JuliaFormatter checks; codecov PR comments
- Docs fragment with per-PR preview deployment and stable/dev documentation
  badges
- Inference fixes for Julia 1.13 (avoid `Core.kwcall` on keyword-only
  constructors in the analyzed paths)

## 0.7.1

- NonconvexCore 1 support
