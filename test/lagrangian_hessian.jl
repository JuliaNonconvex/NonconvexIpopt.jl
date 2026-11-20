@testitem "caller-provided Lagrangian Hessian hook" begin
    using Test, NonconvexCore, NonconvexIpopt

    obj = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    model = Model(obj)
    addvar!(model, [-5.0, -5.0], [5.0, 5.0])
    # The caller supplies the exact Lagrangian Hessian (here the constant
    # Hessian of the unconstrained objective); Ipopt must not fall back to AD.
    calls = Ref(0)
    lh = (x, obj_factor, lambda) -> begin
        calls[] += 1
        return obj_factor .* [2.0 0.0; 0.0 2.0]
    end
    r = optimize(
        model,
        IpoptAlg(),
        [0.0, 0.0];
        options=IpoptOptions((; hessian_approximation="exact")),
        lagrangian_hessian=lh,
    )
    @test maximum(abs.(r.minimizer .- [1.0, 2.0])) < 1e-6
    @test r.minimum < 1e-10
    # the hook must actually be used by eval_h
    @test calls[] > 0
end
