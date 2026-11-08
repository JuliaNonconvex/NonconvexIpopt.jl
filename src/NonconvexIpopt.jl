module NonconvexIpopt

export IpoptAlg, IpoptOptions

using Reexport, Parameters, SparseArrays, Zygote
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!
using NonconvexCore: Solution, ConvergenceState, NoCallback
using NonconvexCore: getnineqconstraints, getneqconstraints
using NonconvexCore: VectorOfFunctions, getdim
import NonconvexCore: optimize!, sparse_jacobian, sparse_hessian, value_jacobian, value_gradient, value_hessian, Workspace
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
    eq = true,
    ineq = true,
    sd = false,
    box = true,
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
function get_ipopt_problem(obj, ineq_constr, eq_constr, x0, xlb, xub, first_order, linear, sparse, symbolic)
    nvars = length(x0)
    if ineq_constr !== nothing
        if symbolic
            ineq_constr = symbolify(ineq_constr, x0; sparse, hessian = !first_order, simplify = true)
            ineq_constr_g = ineq_constr.flat_f.g
            ineq_constr_h = ineq_constr.flat_f.h
            ineqJ0 = ineq_constr_g(x0)
        elseif sparse
            ineq_constr_g = nothing
            ineq_constr_h = nothing
            ineqJ0 = sparse_jacobian(ineq_constr, x0)
        else
            ineq_constr_g = nothing
            ineq_constr_h = nothing
            ineqJ0 = Matrix(value_jacobian(ineq_constr, x0)[2])
        end
        ineq_nconstr, _ = size(ineqJ0)
        ineqJ0_nvalues = nvalues(ineqJ0)
        Joffset = ineqJ0_nvalues
    else
        ineqJ0 = nothing
        ineq_nconstr = 0
        ineqJ0_nvalues = 0
        Joffset = 0
        ineq_constr_g = nothing
        ineq_constr_h = nothing
    end
    if eq_constr !== nothing
        if symbolic
            eq_constr = symbolify(eq_constr, x0; sparse, hessian = !first_order, simplify = true)
            eq_constr_g = eq_constr.flat_f.g
            eq_constr_h = eq_constr.flat_f.h
            eqJ0 = eq_constr_g(x0)
        elseif sparse
            eq_constr_g = nothing
            eq_constr_h = nothing
            eqJ0 = sparse_jacobian(eq_constr, x0)
        else
            eq_constr_g = nothing
            eq_constr_h = nothing
            eqJ0 = Matrix(value_jacobian(eq_constr, x0)[2])
        end
        eq_nconstr, _ = size(eqJ0)
        eqJ0_nvalues = nvalues(eqJ0)
    else
        eqJ0 = nothing
        eqJ0_nvalues = 0
        eq_nconstr = 0
        eq_constr_g = nothing
        eq_constr_h = nothing
    end
    njacvals = ineqJ0_nvalues + eqJ0_nvalues
    # The Jacobian structure is fixed from the initial point; values of later
    # evaluations are extracted from dense Jacobians at these positions.
    jac_rows = zeros(Int, njacvals)
    jac_cols = zeros(Int, njacvals)
    ineqJ0 === nothing || fill_indices!(jac_rows, jac_cols, ineqJ0)
    eqJ0 === nothing ||
        fill_indices!(jac_rows, jac_cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
    jac0 = zeros(njacvals)
    for k = 1:Joffset
        jac0[k] = ineqJ0[jac_rows[k], jac_cols[k]]
    end
    for k = (Joffset+1):njacvals
        jac0[k] = eqJ0[jac_rows[k]-ineq_nconstr, jac_cols[k]]
    end
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
    else
        lag = (factor, y) -> x -> begin
            return factor * obj(x) +
                   _dot(ineq_constr, x, @view(y[1:ineq_nconstr])) +
                   _dot(eq_constr, x, @view(y[ineq_nconstr+1:end]))
        end
    end
    if first_order
        Hnvalues = 0
    else
        L = lag(1.0, ones(ineq_nconstr + eq_nconstr))
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
    cub = zeros(ineq_nconstr + eq_nconstr)

    function eval_g(x::Vector{Float64}, g::Vector{Float64})
        if ineq_constr !== nothing
            g[1:ineq_nconstr] .= ineq_constr(x)
        end
        if eq_constr !== nothing
            g[ineq_nconstr+1:end] .= eq_constr(x)
        end
        return g
    end
    function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
        grad_f .= value_gradient(obj, x)[2]
    end
    function eval_jac_g(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, values::Union{Nothing, Vector{Float64}})
        if values === nothing
            ineqJ0 === nothing || fill_indices!(rows, cols, ineqJ0)
            eqJ0 === nothing || fill_indices!(rows, cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
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
                fill_indices!(rows, cols, HL0)
            elseif symbolic
                _H = nc_hessian(lag(obj_factor, lambda), x)
                HL = LowerTriangular(_H)
                @assert nvalues(HL) == Hnvalues
                values .= 0
                add_values!(values, HL)
            else
                H = value_hessian(lag(obj_factor, lambda), x)
                for k = 1:Hnvalues
                    values[k] = H[h_rows[k], h_cols[k]]
                end
            end
        end
    end
    prob = Ipopt.CreateIpoptProblem(
        nvars, xlb, xub, ineq_nconstr + eq_nconstr, clb, cub,
        nvalues(ineqJ0) + nvalues(eqJ0), Hnvalues, obj,
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
