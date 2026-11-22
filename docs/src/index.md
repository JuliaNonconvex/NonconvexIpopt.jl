# NonconvexIpopt

## Description

[Ipopt](https://coin-or.github.io/Ipopt) is a well known interior point optimizer developed and maintained by COIN-OR. The Julia wrapper of Ipopt is [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl). `Ipopt.jl` is wrapped in `NonconvexIpopt.jl`. `NonconvexIpopt` allows the use of `Ipopt.jl` using the `IpoptAlg` algorithm struct. `IpoptAlg` can be used as a second order optimizer computing the Hessian of the Lagrangian in every iteration. Alternatively, an [l-BFGS approximation](https://en.wikipedia.org/wiki/Limited-memory_BFGS) of the Hessian can be used instead turning `IpoptAlg` into a first order optimizer tha only requires the gradient of the Lagrangian.

See [Quick start](quickstart.md) for the first steps, [Options](options.md) for the algorithm options, [Solver capabilities](capabilities.md) for the model features supported by `IpoptAlg`, [Result](result.md) for the fields of the returned result and [Docstrings](docstrings.md) for the full API reference.
