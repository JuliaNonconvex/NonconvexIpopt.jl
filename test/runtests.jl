using NonconvexIpopt, LinearAlgebra, Test

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

@testset "First order - $first_order, sparse = $sparse, symbolic = $symbolic" for first_order in
                                                                                  [
        true, false
    ],
    sparse in [true, false],
    symbolic in [true, false]

    if !symbolic && sparse
        continue
    end
    options = IpoptOptions(; first_order, sparse, symbolic)
    @testset "Simple constraints" begin
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = IpoptAlg()
        r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Equality constraints" begin
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = IpoptAlg()
        r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Block constraints" begin
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

        alg = IpoptAlg()
        r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Infinite bounds" begin
        @testset "Infinite upper bound" begin
            m = Model(f)
            addvar!(m, [1e-4, 1e-4], [Inf, Inf])
            add_ineq_constraint!(m, x -> g(x, 2, 0))
            add_ineq_constraint!(m, x -> g(x, -1, 1))

            alg = IpoptAlg()
            r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
        @testset "Infinite lower bound" begin
            m = Model(f)
            addvar!(m, [-Inf, 1e-4], [10, 10])
            add_ineq_constraint!(m, x -> g(x, 2, 0))
            add_ineq_constraint!(m, x -> g(x, -1, 1))

            alg = IpoptAlg()
            r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
        @testset "Infinite upper and lower bound" begin
            m = Model(f)
            addvar!(m, [-Inf, 1e-4], [Inf, Inf])
            add_ineq_constraint!(m, x -> g(x, 2, 0))
            add_ineq_constraint!(m, x -> g(x, -1, 1))

            alg = IpoptAlg()
            r = NonconvexIpopt.optimize(m, alg, [1.234, 2.345], options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
    end
end

@testset "IpoptOptions keyword handling" begin
    # Regression tests: these combinations used to silently override user
    # settings.
    o = IpoptOptions(first_order=true, hessian_approximation="exact")
    @test o.nt.hessian_approximation == "exact"
    o = IpoptOptions(first_order=false, hessian_approximation="limited-memory")
    @test o.nt.hessian_approximation == "limited-memory"
    o = IpoptOptions(linear_constraints=true, jac_c_constant="no")
    @test o.nt.jac_c_constant == "no"
    o = IpoptOptions(linear_constraints=false, jac_c_constant="yes")
    @test o.nt.jac_c_constant == "yes"
    # Defaults are still derived from first_order / linear_constraints.
    o = IpoptOptions(first_order=true)
    @test o.nt.hessian_approximation == "limited-memory"
    o = IpoptOptions(first_order=false)
    @test o.nt.hessian_approximation == "exact"
    o = IpoptOptions(linear_constraints=true)
    @test o.nt.jac_c_constant == "yes"
    # sparse without symbolic is not implemented and must fail loudly.
    @test_throws ErrorException IpoptOptions(sparse=true, symbolic=false)
end

@testset "Iteration callback" begin
    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    iterates = Any[]
    r = NonconvexIpopt.optimize(
        m,
        IpoptAlg(),
        [1.234, 2.345],
        options=IpoptOptions(),
        callback=s -> push!(iterates, (x=copy(s.x), f=s.f, infeas=s.convstate.infeas)),
    )
    @test length(iterates) > 0
    @test all(s -> isfinite(s.f), iterates)
    # The callback sees the sequence of Ipopt iterates; the final iterate
    # is the accepted solution.
    @test abs(iterates[end].f - r.minimum) < 1e-4
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
end

using Aqua, JET
Aqua.test_all(NonconvexIpopt)
JET.test_package(NonconvexIpopt; target_modules = (NonconvexIpopt,))
