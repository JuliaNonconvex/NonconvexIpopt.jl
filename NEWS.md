# NEWS

## 0.7.1

- `optimize` now accepts the generic `callback` keyword. The callback runs
  once per Ipopt iteration through Ipopt's intermediate callback and receives
  a `Solution` carrying the current iterate (from
  `Ipopt.GetIpoptCurrentIterate`), the objective value, the constraint values
  and multipliers, and the primal/dual infeasibility in the convergence
  state.
