module NonconvexIpopt

export IpoptAlg, IpoptOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult, symbolify
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!
using NonconvexCore: Solution, ConvergenceState, NoCallback
using NonconvexCore: getnineqconstraints, getneqconstraints
using NonconvexCore: VectorOfFunctions, getdim
import NonconvexCore: optimize!, sparse_jacobian, sparse_hessian, value_jacobian, value_gradient, value_hessian, Workspace
using NonconvexCore: getinputunflatten, getrawfunction
using Ipopt

@params struct IpoptOptions
    nt::NamedTuple
end
function IpoptOptions(;
    first_order = true,
    linear_constraints = false,
    hessian_approximation = nothing,
    sparse = false,
    symbolic = false,
    kwargs...,
)
    _hessian_approximation =
        hessian_approximation === nothing ? (first_order ? "limited-memory" : "exact") :
        hessian_approximation
    jac_c_constant =
        haskey(kwargs, :jac_c_constant) ? kwargs[:jac_c_constant] :
        (linear_constraints ? "yes" : "no")
    jac_d_constant =
        haskey(kwargs, :jac_d_constant) ? kwargs[:jac_d_constant] :
        (linear_constraints ? "yes" : "no")
    kwargs = (;
        kwargs...,
        hessian_approximation = _hessian_approximation,
        jac_c_constant = jac_c_constant,
        jac_d_constant = jac_d_constant,
        sparse,
        symbolic,
    )
    return IpoptOptions(kwargs)
end

@params mutable struct IpoptWorkspace <: Workspace
    model::VecModel
    problem::Ipopt.IpoptProblem
    x0::AbstractVector
    options::IpoptOptions
    counter::Base.RefValue{Int}
    callback::Function
end
function IpoptWorkspace(
    model::VecModel, x0::AbstractVector = getinit(model);
    options = IpoptOptions(),
    callback::Function = NoCallback(),
    kwargs...,
)
    problem, counter = get_ipopt_problem(
        model, copy(x0),
        options.nt.hessian_approximation == "limited-memory",
        options.nt.jac_c_constant == "yes" && options.nt.jac_d_constant == "yes",
        options.nt.sparse,
        options.nt.symbolic,
    )
    return IpoptWorkspace(model, problem, copy(x0), options, counter, callback)
end
@params struct IpoptResult <: AbstractResult
    minimizer
    minimum
    problem
    status
    fcalls::Int
end

function optimize!(workspace::IpoptWorkspace)
    @unpack model, problem, options, counter, x0, callback = workspace
    problem.x .= x0
    counter[] = 0
    foreach(keys(options.nt)) do k
        if k != :sparse && k != :symbolic
            v = options.nt[k]
            addOption(problem, string(k), v)
        end
    end
    if callback !== NoCallback()
        setup_intermediate_callback!(workspace)
    end
    solvestat = Ipopt.IpoptSolve(problem)
    return IpoptResult(
        copy(problem.x), getobjective(model)(problem.x),
        problem, solvestat, counter[]
    )
end

"""
    setup_intermediate_callback!(workspace)

Registers Ipopt's intermediate callback so the Nonconvex `callback` is invoked
once per Ipopt iteration with a `Solution` carrying the current iterate, the
constraint values, the constraint multipliers, the objective value, and the
primal/dual infeasibility in the convergence state.
"""
function setup_intermediate_callback!(workspace::IpoptWorkspace)
    @unpack model, problem, x0, callback = workspace
    n = length(x0)
    m = getnineqconstraints(model) + getneqconstraints(model)
    x = zeros(n)
    z_L = zeros(n)
    z_U = zeros(n)
    g = zeros(m)
    λ = zeros(m)
    Ipopt.SetIntermediateCallback(problem, (
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ) -> begin
        Ipopt.GetIpoptCurrentIterate(problem, false, n, x, z_L, z_U, m, g, λ)
        callback(ipopt_solution(x, g, λ, obj_value, inf_pr, inf_du))
        return true
    end)
end

function ipopt_solution(x::AbstractVector, g::AbstractVector, λ::AbstractVector, obj_value::Real, inf_pr::Real, inf_du::Real)
    n = length(x)
    m = length(g)
    convstate = ConvergenceState{Float64}()
    convstate.kkt_residual = inf_du
    convstate.ipopt_residual = inf_du
    convstate.infeas = inf_pr
    return Solution(
        copy(x),
        copy(x),
        λ,
        obj_value,
        obj_value,
        zeros(n),
        g,
        zeros(m, n),
        convstate,
    )
end
function addOption(prob, name, val::Int)
    return Ipopt.AddIpoptIntOption(prob, name, val)
end
function addOption(prob, name, val::String)
    return Ipopt.AddIpoptStrOption(prob, name, val)
end
function addOption(prob, name, val)
    return Ipopt.AddIpoptNumOption(prob, name, val)
end

struct IpoptAlg <: AbstractOptimizer end

NonconvexCore.solver_capabilities(::IpoptAlg) = ConstraintCapabilities(
    linear_eq = true,
    nonlinear_eq = true,
    linear_ineq = true,
    nonlinear_ineq = true,
    linear_sd = false,
    nonlinear_sd = false,
    box = true,
    continuous = true,
    binary = false,
    integer = false,
    infinite_bounds = true,
)

function Workspace(model::VecModel, optimizer::IpoptAlg, x0::AbstractVector; kwargs...,)
    return IpoptWorkspace(model, x0; kwargs...)
end

function get_ipopt_problem(model::VecModel, x0::AbstractVector, first_order::Bool, linear::Bool, sparse::Bool, symbolic::Bool)
    eq = if length(model.eq_constraints.fs) == 0
        nothing
    else
        model.eq_constraints
    end
    ineq = if length(model.ineq_constraints.fs) == 0
        nothing
    else
        model.ineq_constraints
    end
    _obj = if symbolic
        symbolify(getobjective(model), x0; sparse, hessian = !first_order, simplify = true)
    # elseif sparse
    #     sparsify(getobjective(model), x0; hessian = !first_order)
    else
        getobjective(model)
    end
    obj = CountingFunction(_obj)
    return get_ipopt_problem(
        obj,
        ineq,
        eq,
        x0,
        getmin(model),
        getmax(model),
        first_order,
        linear,
        sparse,
        symbolic,
    ), obj.counter
end
# The single Nonlinear evaluator of a JuMP-converted model, if the objective
# and every constraint belong to it; otherwise `nothing`.
function _jump_evaluator(obj, ineq_constr, eq_constr)
    objraw = obj isa CountingFunction ? obj.f : obj
    objraw = objraw isa NonconvexCore.Objective ? NonconvexCore.getfunction(objraw) : objraw
    objraw = objraw isa NonconvexCore.FlattenedFunction ? NonconvexCore.getrawfunction(objraw) : objraw
    objraw isa NonconvexCore.JuMPModelObjective || return nothing
    m = objraw.m
    cs = (
        (ineq_constr === nothing ? () : ineq_constr.fs)...,
        (eq_constr === nothing ? () : eq_constr.fs)...,
    )
    for c in cs
        raw = c.f isa NonconvexCore.FlattenedFunction ? NonconvexCore.getrawfunction(c.f) : c.f
        raw = raw isa FunctionWrapper ? NonconvexCore.getfunction(raw) : raw
        (raw isa NonconvexCore.JuMPModelIneq || raw isa NonconvexCore.JuMPModelEq) || return nothing
        raw.m === m || return nothing
    end
    return m
end

function get_ipopt_problem(obj, ineq_constr, eq_constr, x0, xlb, xub, first_order, linear, sparse, symbolic)
    nvars = length(x0)
    # Batched model pieces for the non-symbolic path: the raw user-facing
    # functions and a single input unflattening, so that every point rebuilds
    # the user-facing structure once and every derivative of the model shares
    # one prepared ReverseDiff tape.
    unx = nothing
    ineq_cs = ()
    eq_cs = ()
    rawpieces(c) = c.f isa NonconvexCore.FlattenedFunction ?
                   (NonconvexCore.getrawfunction(c.f), NonconvexCore.getinputunflatten(c.f)) :
                   (c.f, t -> (t,))
    if ineq_constr !== nothing
        raw0, unx = rawpieces(ineq_constr.fs[1])
        ineq_cs = map(ineq_constr.fs) do c
            raw, _ = rawpieces(c)
            (raw, c.rhs, c.dim)
        end
    end
    if eq_constr !== nothing
        if unx === nothing
            _, unx = rawpieces(eq_constr.fs[1])
        end
        eq_cs = map(eq_constr.fs) do c
            raw, _ = rawpieces(c)
            (raw, c.rhs, c.dim)
        end
    end
    ineq_nconstr = sum(c -> c[3], ineq_cs; init = 0)
    eq_nconstr = sum(c -> c[3], eq_cs; init = 0)
    nconstr = ineq_nconstr + eq_nconstr
    # A JuMP-converted model provides values, Jacobian and Lagrangian
    # Hessian directly from its single Nonlinear evaluator.
    jump = symbolic ? nothing : _jump_evaluator(obj, ineq_constr, eq_constr)
    if symbolic
        if ineq_constr !== nothing
            ineq_constr = symbolify(ineq_constr, x0; sparse, hessian = !first_order, simplify = true)
            ineq_constr_g = ineq_constr.flat_f.g
            ineq_constr_h = ineq_constr.flat_f.h
        else
            ineq_constr_g = nothing
            ineq_constr_h = nothing
        end
        if eq_constr !== nothing
            eq_constr = symbolify(eq_constr, x0; sparse, hessian = !first_order, simplify = true)
            eq_constr_g = eq_constr.flat_f.g
            eq_constr_h = eq_constr.flat_f.h
        else
            eq_constr_g = nothing
            eq_constr_h = nothing
        end
        ineqJ0 = ineq_constr_g === nothing ? nothing : ineq_constr_g(x0)
        eqJ0 = eq_constr_g === nothing ? nothing : eq_constr_g(x0)
        ineqJ0_nvalues = nvalues(ineqJ0)
        eqJ0_nvalues = nvalues(eqJ0)
        Joffset = ineqJ0_nvalues
    else
        ineqJ0 = ineq_constr === nothing ? nothing :
                 (sparse ? sparse_jacobian(ineq_constr, x0) : Matrix(value_jacobian(ineq_constr, x0)[2]))
        eqJ0 = eq_constr === nothing ? nothing :
               (sparse ? sparse_jacobian(eq_constr, x0) : Matrix(value_jacobian(eq_constr, x0)[2]))
        ineqJ0_nvalues = nvalues(ineqJ0)
        eqJ0_nvalues = nvalues(eqJ0)
        Joffset = ineqJ0_nvalues
    end
    if jump !== nothing
        ineqJ0 = nothing
        eqJ0 = nothing
        Joffset = 0
        njacvals = length(NonconvexCore.jump_jacobian_structure(jump)[1])
    else
        njacvals = ineqJ0_nvalues + eqJ0_nvalues
    end
    # The Jacobian structure is fixed from the initial point.
    jac_rows = zeros(Int, njacvals)
    jac_cols = zeros(Int, njacvals)
    if jump !== nothing
        jac_rows0, jac_cols0 = NonconvexCore.jump_jacobian_structure(jump)
        jac_rows .= jac_rows0
        jac_cols .= jac_cols0
    else
        ineqJ0 === nothing || fill_indices!(jac_rows, jac_cols, ineqJ0)
        eqJ0 === nothing ||
            fill_indices!(jac_rows, jac_cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
    end
    jac0 = zeros(njacvals)
    if jump !== nothing
        NonconvexCore.jump_jacobian!(jump, jac0, x0)
    else
        for k = 1:Joffset
            jac0[k] = ineqJ0[jac_rows[k], jac_cols[k]]
        end
        for k = (Joffset+1):njacvals
            jac0[k] = eqJ0[jac_rows[k]-ineq_nconstr, jac_cols[k]]
        end
    end
    # For the non-symbolic second-order path the Lagrangian is a single
    # closure over mutable multipliers: the sparse Hessian preparation
    # (sparsity detection and coloring) happens once at construction and is
    # reused for every evaluation.
    lag_refs = (factor = Ref(1.0), lambda = Ref(ones(nconstr)))
    if symbolic
        lag = (factor, y) -> SparseLagrangian(
            obj,
            x -> Zygote.gradient(obj, x)[1],
            x -> Zygote.hessian(obj, x),
            factor,
            ineq_constr,
            ineq_constr_g,
            ineq_constr_h,
            eq_constr,
            eq_constr_g,
            eq_constr_h,
            y[1:ineq_nconstr],
            y[ineq_nconstr+1:end],
        )
        lagf = nothing
    else
        lagf = x -> begin
            return lag_refs.factor[] * obj(x) +
                   _dot(ineq_constr, x, @view(lag_refs.lambda[][1:ineq_nconstr])) +
                   _dot(eq_constr, x, @view(lag_refs.lambda[][ineq_nconstr+1:end]))
        end
    end
    if first_order
        Hnvalues = 0
    elseif jump !== nothing
        h_rows0, h_cols0 = NonconvexCore.jump_hessian_structure(jump)
        Hnvalues = length(h_rows0)
        h_rows = Int.(h_rows0)
        h_cols = Int.(h_cols0)
        HL0 = nothing
    else
        if symbolic
            L = lag(1.0, ones(nconstr))
        else
            lag_refs.factor[] = 1.0
            lag_refs.lambda[] = ones(nconstr)
            L = lagf
        end
        # The Hessian structure is detected once at `x0` with all constraint
        # multipliers active so that it contains every possible entry.
        protoH = symbolic ? nc_hessian(L, x0) :
                 sparse ? sparse_hessian(L, x0) : nc_hessian(L, x0)
        HL0 = LowerTriangular(protoH)
        Hnvalues = nvalues(HL0)
        h_rows = zeros(Int, Hnvalues)
        h_cols = zeros(Int, Hnvalues)
        fill_indices!(h_rows, h_cols, HL0)
    end

    clb = [fill(-Inf, ineq_nconstr); zeros(eq_nconstr)]
    cub = zeros(nconstr)

    function eval_g(x::Vector{Float64}, g::Vector{Float64})
        if jump !== nothing
            NonconvexCore.jump_constraint_values!(jump, g, x)
        elseif symbolic
            ineq_constr === nothing || (g[1:ineq_nconstr] .= ineq_constr(x))
            eq_constr === nothing || (g[ineq_nconstr+1:end] .= eq_constr(x))
        else
            d = unx(x)
            off = 0
            for (raw, rhs, dim) in ineq_cs
                g[off+1:off+dim] .= raw(d...) .- rhs
                off += dim
            end
            for (raw, rhs, dim) in eq_cs
                g[off+1:off+dim] .= raw(d...) .- rhs
                off += dim
            end
        end
        return g
    end
    function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
        if jump !== nothing
            NonconvexCore.jump_objective_gradient!(jump, grad_f, x)
        else
            grad_f .= value_gradient(obj, x)[2]
        end
    end
    function eval_jac_g(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, values::Union{Nothing, Vector{Float64}})
        if values === nothing
            if jump !== nothing
                rows .= jac_rows
                cols .= jac_cols
            else
                ineqJ0 === nothing || fill_indices!(rows, cols, ineqJ0)
                eqJ0 === nothing || fill_indices!(rows, cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
            end
        elseif jump !== nothing
            NonconvexCore.jump_jacobian!(jump, values, x)
        elseif linear
            values .= jac0
        elseif symbolic
            values .= 0
            ineq_constr === nothing || add_values!(values, ineq_constr_g(x))
            eq_constr === nothing || add_values!(values, eq_constr_g(x), offset = Joffset)
    else
        if ineq_constr !== nothing
            J = Matrix(value_jacobian(ineq_constr, x)[2])
            for k = 1:Joffset
                values[k] = J[jac_rows[k], jac_cols[k]]
            end
        end
        if eq_constr !== nothing
            J = Matrix(value_jacobian(eq_constr, x)[2])
            for k = (Joffset+1):njacvals
                values[k] = J[jac_rows[k]-ineq_nconstr, jac_cols[k]]
            end
        end
    end
    end

    if first_order
        eval_h = (x...) -> 0.0
    else
        eval_h = function (x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Union{Nothing, Vector{Float64}})
            if values === nothing
                if jump !== nothing
                    rows .= h_rows
                    cols .= h_cols
                else
                    fill_indices!(rows, cols, HL0)
                end
            elseif jump !== nothing
                NonconvexCore.jump_hessian_lagrangian!(jump, values, x, obj_factor, lambda)
            elseif symbolic
                _H = nc_hessian(lag(obj_factor, lambda), x)
                HL = LowerTriangular(_H)
                @assert nvalues(HL) == Hnvalues
                values .= 0
                add_values!(values, HL)
            else
                lag_refs.factor[] = obj_factor
                lag_refs.lambda[] = lambda
                H = sparse ? sparse_hessian(lagf, x) : nc_hessian(lagf, x)
                HL = LowerTriangular(H)
                @assert nvalues(HL) == Hnvalues
                values .= 0
                add_values!(values, HL)
            end
        end
    end
    prob = Ipopt.CreateIpoptProblem(
        nvars, xlb, xub, ineq_nconstr + eq_nconstr, clb, cub,
        jump !== nothing ? njacvals : nvalues(ineqJ0) + nvalues(eqJ0), Hnvalues, obj,
        eval_g, eval_grad_f, eval_jac_g, eval_h,
    )
    prob.x = x0
    return prob
end

_dot(f, x, y) = f(x)' * y
_dot(::Nothing, ::Any, ::Any) = 0.0
function _dot(f::VectorOfFunctions, x, y)
    # Accumulate one term per function instead of calling `f(x)` as a whole so
    # that reverse-mode backends differentiate each function separately.
    s = 0.0
    offset = 0
    for fi in f.fs
        dim = getdim(fi)
        s += dot(fi(x), @view(y[offset+1:offset+dim]))
        offset += dim
    end
    return s
end
_reshape_dot(::Nothing, x, ::Any) = zeros(length(x), length(x))
_reshape_dot(f, x, y) = reshape(reshape(f(x), length(y), length(x)^2)' * y, length(x), length(x))

struct SparseLagrangian{F0, J0, H0, F, G1, J1, H1, G2, J2, H2, L1, L2}
    f::F0
    fJ::J0
    fH::H0
    factor::F
    ineq::G1
    ineqJ::J1
    ineqH::H1
    eq::G2
    eqJ::J2
    eqH::H2
    λ1::L1
    λ2::L2
end
function (l::SparseLagrangian)(x)
    return l.factor * l.f(x) + _dot(l.ineq, x, l.λ1) + _dot(l.eq, x, l.λ2)
end
function nc_gradient(l::SparseLagrangian, x)
    return l.factor * l.fJ(x) + _dot(l.ineqJ, x, l.λ1) + _dot(l.eqJ, x, l.λ2)
end
nc_gradient(l, x) = Zygote.gradient(l, x)[1]
function nc_hessian(l::SparseLagrangian, x)
    return l.factor * l.fH(x) + _reshape_dot(l.ineqH, x, l.λ1) + _reshape_dot(l.eqH, x, l.λ2)
end
nc_hessian(l, x) = Zygote.hessian(l, x)

end
