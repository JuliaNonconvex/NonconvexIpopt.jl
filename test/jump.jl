@testitem "JuMP-converted models use the evaluator fast path" begin
    using NonconvexIpopt, NonconvexCore, JuMP, Ipopt, LinearAlgebra, Test

    # A small nonlinear JuMP model exercising both nonlinear and quadratic
    # constraints, plus a bound.
    function jump_model()
        m = JuMP.Model()
        JuMP.@variable(m, x[1:3])
        JuMP.@constraint(m, x[1] + x[2] == 1.0)
        JuMP.@constraint(m, x[2]^2 + x[3]^2 <= 2.0)
        JuMP.@constraint(m, x[1] + cos(x[2] * x[3]) <= 2.5)
        JuMP.@constraint(m, sin(x[1]) + x[3] >= -1.0)
        JuMP.@objective(m, Min, (x[1] - 1)^2 + (x[2] - 2)^2 + x[3]^2)
        return m
    end

    jm = jump_model()
    dm = NonconvexCore.DictModel(jm)

    # Sanity: the converted objective and constraints share the single
    # Nonlinear evaluator of the converted model.
    obj = NonconvexCore.getobjective(dm)
    objraw = if obj.f isa NonconvexCore.FlattenedFunction
        NonconvexCore.getrawfunction(obj.f)
    else
        obj.f
    end
    @test objraw isa NonconvexCore.JuMPModelObjective

    # Regression: the JuMP-converted path must not fall back to computing the
    # dense AD Jacobian. Spy on the AD entry points across a full solve: on
    # the fast path neither is ever called (values, gradient, Jacobian and
    # Hessian all come from the evaluator).
    calls = Ref(0)
    orig_vj = NonconvexIpopt._value_jacobian
    orig_sj = NonconvexIpopt._sparse_jacobian
    spy_vj = function (args...; kwargs...)
        calls[] += 1
        return orig_vj(args...; kwargs...)
    end
    spy_sj = function (args...; kwargs...)
        calls[] += 1
        return orig_sj(args...; kwargs...)
    end
    NonconvexIpopt._value_jacobian = spy_vj
    NonconvexIpopt._sparse_jacobian = spy_sj
    try
        x0 = NonconvexCore.getinit(dm)
        r = NonconvexIpopt.optimize(
            dm, IpoptAlg(), x0; options=IpoptOptions(; first_order=false)
        )
        @test calls[] == 0
        @test startswith(r.status, "Solve_Succeeded")

        # Same solution as the direct JuMP+Ipopt solve.
        JuMP.set_optimizer(jm, Ipopt.Optimizer)
        JuMP.set_optimizer_attribute(jm, "print_level", 0)
        JuMP.optimize!(jm)
        @test JuMP.termination_status(jm) == JuMP.LOCALLY_SOLVED
        @test abs(JuMP.objective_value(jm) - r.minimum) < 1e-6
    finally
        NonconvexIpopt._value_jacobian = orig_vj
        NonconvexIpopt._sparse_jacobian = orig_sj
    end
end

@testitem "JuMP-converted fast path solves correctly" begin
    using NonconvexIpopt, NonconvexCore, JuMP, Ipopt, LinearAlgebra, Test

    m = JuMP.Model()
    JuMP.@variable(m, x[1:2])
    JuMP.@constraint(m, x[1]^2 + x[2] <= 3.0)
    JuMP.@constraint(m, x[1] + exp(x[2] / 10) >= 0.5)
    JuMP.@objective(m, Min, (x[1] - 2)^2 + 4(x[2] + 1)^2)

    dm = NonconvexCore.DictModel(m)
    r = NonconvexIpopt.optimize(
        dm, IpoptAlg(), NonconvexCore.getinit(dm); options=IpoptOptions(; first_order=false)
    )
    @test startswith(r.status, "Solve_Succeeded")
    @test r.minimum ≈ 0.0 atol=1e-5
end
