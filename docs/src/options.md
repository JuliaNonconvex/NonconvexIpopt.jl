# Options

The options keyword argument to the `optimize` function shown above must be an instance of the `IpoptOptions` struct when the algorihm is an `IpoptAlg`. To specify options use keyword arguments in the constructor of `IpoptOptions`, e.g:
```julia
options = IpoptOptions(first_order = false, tol = 1e-4, sparse = false, symbolic = false)
```
There are 4 important and special options:
- `first_order`: `true` by default. When `first_order` is `true`, the first order Ipopt algorithm will be used. And when it is `false`, the second order Ipopt algorithm will be used.
- `symbolic`: `false` by default. When `symbolic` is set to `true`, the gradients, Jacobians and Hessians of the objective, constraint and Lagrangian functions will be calculated using symbolic differentiation from [`Symbolics.jl`](https://github.com/JuliaSymbolics/Symbolics.jl). This is the same approach used by `symbolify` which is described in the [symbolic differentiation section](../core/derivatives/backends.md) in the documentation.
- `sparse`: `false` by default. When `sparse` is set to `true`, the gradients, Jacobians and Hessians of the objective, constraint and Lagrangian functions will be treated as sparse vectors/matrices. When combined with `symbolic = true`, the output of symbolic differentiation will be a sparse vector/matrix, akin to setting `sparse = true` in the `symbolify` function discussed in [symbolic differentiation section](../core/derivatives/backends.md) in the documentation. When used alone with `symbolic = false`, [`SparseDiffTools.jl`](https://github.com/JuliaDiff/SparseDiffTools.jl) is used instead for the differentiation and `Symbolics` is only used to get the sparsity pattern, much like how `sparsify` works. For more details on `sparsify` and the way `SparseDiffTools` works, see the [sparsity section](../core/derivatives/sparse.md) in the documentation is used instead.
- `linear_constraints`:  `false` by default. When `linear_constraints` is `true`, the Jacobian of the constraints will be computed and sparsified once at the beginning. When it is `false`, dense Jacobians will be computed in every iteration.

> Note that there is no need to use `sparsify` or `symbolify` on the model or functions before optimizing it with an `IpoptAlg`. Setting the `sparse` and `symbolic` options above are enough to trigger the symbolic differentiation and/or sparsity exploitation.

All the other options that can be set can be found on the [Ipopt options](https://coin-or.github.io/Ipopt/OPTIONS.html) section of Ipopt's documentation.
