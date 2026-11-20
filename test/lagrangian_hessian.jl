@testitem "caller-provided Lagrangian Hessian hook" begin
    using Test, NonconvexCore, NonconvexIpopt

    obj = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    model = Model(obj)
    addvar!(model, [-5.0, -5.0], [5.0, 5.0])
    # The caller supplies the exact Lagrangian Hessian (here the constant
    # Hessian of the unconstrained objective); Ipopt must not fall back to AD.
    lh = (x, obj_factor, lambda) -> obj_factor .* [2.0 0.0; 0.0 2.0]
    r = optimize(
        model,
        IpoptAlg(),
        [0.0, 0.0];
        options = IpoptOptions(hessian_approximation = "exact"),
        lagrangian_hessian = lh,
    )
    @test maximum(abs.(r.minimizer .- [1.0, 2.0])) < 1e-6
    @test r.minimum < 1e-10
end
