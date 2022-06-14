@params struct IpoptOptions
    nt::NamedTuple
end
function IpoptOptions(;
    first_order = true,
    linear_constraints = false,
    hessian_approximation = first_order ? "limited-memory" : "exact",
    kwargs...,
)
    kwargs = if linear_constraints
        (;kwargs..., hessian_approximation = hessian_approximation, jac_c_constant = "yes", jac_d_constant = "yes")
    else
        (;kwargs..., hessian_approximation = hessian_approximation, jac_c_constant = "no", jac_d_constant = "no")
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
        v = options.nt[k]
        addOption(problem, string(k), v)
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

function get_ipopt_problem(model::VecModel, x0::AbstractVector, first_order::Bool, linear::Bool)
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
    obj = CountingFunction(getobjective(model))
    return get_ipopt_problem(
        obj,
        ineq,
        eq,
        x0,
        getmin(model),
        getmax(model),
        first_order,
        linear,
    ), obj.counter
end
function get_ipopt_problem(obj, ineq_constr, eq_constr, x0, xlb, xub, first_order, linear)
    nvars = length(x0)
    if ineq_constr !== nothing
        ineqJ0 = sparse_jacobian(ineq_constr, x0)
        ineqJ0 = linear ? sparse(ineqJ0) : ineqJ0
        ineq_nconstr, _ = size(ineqJ0)
        Joffset = nvalues(ineqJ0)
    else
        ineqJ0 = nothing
        ineq_nconstr = 0
        Joffset = 0
    end
    if eq_constr !== nothing
        eqJ0 = sparse_jacobian(eq_constr, x0)
        eqJ0 = linear ? sparse(eqJ0) : eqJ0
        eq_nconstr, _ = size(eqJ0)
    else
        eqJ0 = nothing
        eq_nconstr = 0
    end
    lag(factor, y) = x -> begin
        return factor * obj(x) + 
            _dot(ineq_constr, x, @view(y[1:ineq_nconstr])) + 
            _dot(eq_constr, x, @view(y[ineq_nconstr+1:end]))
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
                ineqJ = linear ? ineqJ0 : sparse_jacobian(ineq_constr, x)
                add_values!(values, ineqJ)
            end
            if eq_constr !== nothing
                eqJ = linear ? eqJ0 : sparse_jacobian(eq_constr, x)
                add_values!(values, eqJ, offset = Joffset)
            end
        end
    end

    if first_order
        eval_h = (x...) -> 0.0
        Hnvalues = 0
    else
        HL0 = LowerTriangular(
            sparse_hessian(
                lag(1.0, ones(ineq_nconstr + eq_nconstr)),
                x0,
            ),
        )
        eval_h = function (x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Union{Nothing, Vector{Float64}})
            if values === nothing
                fill_indices!(rows, cols, HL0)
            else
                HL = LowerTriangular(
                    sparse_hessian(lag(obj_factor, lambda), x),
                )
                values .= 0
                add_values!(values, HL)
            end
        end
        Hnvalues = nvalues(HL0)
    end
    _obj = function (x)
        try
            return obj(x)
        catch
            return Inf
        end
    end
    prob = Ipopt.CreateIpoptProblem(
        nvars, xlb, xub, ineq_nconstr + eq_nconstr, clb, cub,
        nvalues(ineqJ0) + nvalues(eqJ0), Hnvalues, _obj,
        eval_g, eval_grad_f, eval_jac_g, eval_h,
    )
    prob.x = x0
    return prob
end
