<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexIpopt.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexIpopt.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexIpopt.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexIpopt.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexIpopt.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexIpopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexIpopt.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexIpopt.jl

Interior point optimization with Ipopt.jl

First and second order interior point optimization with sparse and symbolic differentiation options.

NonconvexIpopt.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
using NonconvexIpopt

alg = IpoptAlg()
options = IpoptOptions()
result = optimize(model, alg, x0, options = options)
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexIpopt.jl/stable](https://julianonconvex.github.io/NonconvexIpopt.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexIpopt.jl/dev](https://julianonconvex.github.io/NonconvexIpopt.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexIpopt")
```
