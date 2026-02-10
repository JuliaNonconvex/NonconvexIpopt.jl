module NonconvexIpopt

export IpoptAlg, IpoptOptions

using Reexport, Parameters, SparseArrays, Zygote, NonconvexUtils
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: nvalues, fill_indices!, add_values!
import NonconvexCore: optimize!, sparse_jacobian, sparse_hessian, Workspace
using Ipopt

@params struct IpoptOptions
    nt::NamedTuple
end
function IpoptOptions(;
    first_order = true,
    linear_constraints = false,
    hessian_approximation = first_order ? "limited-memory" : "exact",
    sparse = false,
    symbolic = false,
    kwargs...,
)
    kwargs = if linear_constraints
        (; kwargs..., hessian_approximation, jac_c_constant = "yes", jac_d_constant = "yes", sparse, symbolic)
    else
        (; kwargs..., hessian_approximation, jac_c_constant = "no", jac_d_constant = "no", sparse, symbolic)
    end
    return IpoptOptions(kwargs)
end

@params mutable struct IpoptWorkspace <: Workspace
    model::VecModel
    problem::Ipopt.IpoptProblem
    x0::AbstractVector
    options::IpoptOptions
    counter::Base.RefValue{Int}
end
function IpoptWorkspace(
    model::VecModel, x0::AbstractVector = getinit(model);
    options = IpoptOptions(), kwargs...,
)
    problem, counter = get_ipopt_problem(
        model, copy(x0),
        options.nt.hessian_approximation == "limited-memory",
        options.nt.jac_c_constant == "yes" && options.nt.jac_d_constant == "yes",
        options.nt.sparse,
        options.nt.symbolic,
    )
    return IpoptWorkspace(model, problem, copy(x0), options, counter)
end
@params struct IpoptResult <: AbstractResult
    minimizer
    minimum
    problem
    status
    fcalls::Int
end

function optimize!(workspace::IpoptWorkspace)
    @unpack model, problem, options, counter, x0 = workspace
    problem.x .= x0
    counter[] = 0
    foreach(keys(options.nt)) do k
        if k != :sparse && k != :symbolic
            v = options.nt[k]
            addOption(problem, string(k), v)
        end
    end
    solvestat = Ipopt.IpoptSolve(problem)
    return IpoptResult(
        copy(problem.x), getobjective(model)(problem.x),
        problem, solvestat, counter[]
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
        # elseif sparse
        #     ineq_constr = sparsify(ineq_constr, x0; hessian = !first_order)
        #     ineq_constr_g = ineq_constr.flat_f.J
        #     ineq_constr_h = ineq_constr.flat_f.H
        else
            ineq_constr_g = x -> Zygote.jacobian(ineq_constr, x)[1]
            ineq_constr_h = nothing
        end
        ineqJ0 = ineq_constr_g(x0)
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
        # elseif sparse
        #     eq_constr = sparsify(eq_constr, x0; hessian = !first_order)
        #     eq_constr_g = eq_constr.flat_f.J
        #     eq_constr_h = eq_constr.flat_f.H
        else
            eq_constr_g = x -> Zygote.jacobian(eq_constr, x)[1]
            eq_constr_h = nothing
        end
        eqJ0 = eq_constr_g(x0)
        eqJ0_nvalues = nvalues(eqJ0)
        eq_nconstr, _ = size(eqJ0)
    else
        eqJ0 = nothing
        eqJ0_nvalues = 0
        eq_nconstr = 0
        eq_constr_g = nothing
        eq_constr_h = nothing
    end
    if sparse || symbolic
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
            return factor * obj(x) + _dot(ineq_constr, x, y[1:ineq_nconstr]) + _dot(eq_constr, x, y[ineq_nconstr+1:end])
        end
    end
    if first_order
        Hnvalues = 0
    else
        L = lag(1.0, ones(ineq_nconstr + eq_nconstr))
        protoH = nc_hessian(L, x0)
        HL0 = LowerTriangular(protoH)
        Hnvalues = nvalues(HL0)
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
        grad = Zygote.gradient(obj, x)[1]
        if grad === nothing
            grad_f .= 0
        else
            grad_f .= grad
        end
    end
    function eval_jac_g(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, values::Union{Nothing, Vector{Float64}})
        if values === nothing
            ineqJ0 === nothing || fill_indices!(rows, cols, ineqJ0)
            eqJ0 === nothing || fill_indices!(rows, cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
        else
            values .= 0
            if ineq_constr !== nothing
                ineqJ = linear ? ineqJ0 : ineq_constr_g(x)
                @assert nvalues(ineqJ) == ineqJ0_nvalues
                add_values!(values, ineqJ)
            end
            if eq_constr !== nothing
                eqJ = linear ? eqJ0 : eq_constr_g(x)
                @assert nvalues(eqJ) == eqJ0_nvalues
                add_values!(values, eqJ, offset = Joffset)
            end
        end
    end

    if first_order
        eval_h = (x...) -> 0.0
    else
        eval_h = function (x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Union{Nothing, Vector{Float64}})
            if values === nothing
                fill_indices!(rows, cols, HL0)
            else
                _H = nc_hessian(lag(obj_factor, lambda), x)
                HL = LowerTriangular(_H)
                @assert nvalues(HL) == Hnvalues
                values .= 0
                add_values!(values, HL)
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
