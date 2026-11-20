# Result

`optimize` returns an [`IpoptResult`](@ref) with the `GenericResult`-style fields:

- `minimizer`, `minimum`: the solution found by Ipopt,
- `problem`: the underlying [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) problem object (the raw Ipopt handle),
- `status`: the Ipopt solve status,
- `fcalls`: the number of objective evaluations performed during the call.

A `callback` passed to `optimize` is invoked with the intermediate iterates; see [the intermediate callback guide](../core/criteria.md) for details.
