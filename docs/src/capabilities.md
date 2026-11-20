# Solver capabilities

[`IpoptAlg`](@ref) declares its capabilities through `solver_capabilities`:

- `linear_eq = true`, `nonlinear_eq = true`: equality constraints of any linearity,
- `linear_ineq = true`, `nonlinear_ineq = true`,
- `box = true`,
- `continuous = true`,
- `infinite_bounds = true`,
- semidefinite constraints (`linear_sd`, `nonlinear_sd`), binary variables and integer variables are **not** supported: models with those features error with an informative message before optimization starts.

The declaration is checked against the model via `checksubproblem!` before the solve, so unsupported model features are caught early.
