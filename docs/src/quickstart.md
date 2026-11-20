# Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using Ipopt.
```julia
using NonconvexIpopt

alg = IpoptAlg()
options = IpoptOptions()
result = optimize(model, alg, x0, options = options)
```

## Construct an instance

To construct an instance of the Ipopt algorithm, use:
```julia
alg = IpoptAlg()
```

`IpoptAlg` can be used as a sub-solver of the meta-algorithms in the ecosystem, e.g. `BayesOptAlg(IpoptAlg())` for the surrogate solves of Bayesian optimization.
