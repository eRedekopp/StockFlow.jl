module Semantics

export ModelSemantics,
    ODESemantics,
    Interpretation,
    SimulationInterpretation,
    interpret,
    timepoints,
    traj

using StockFlow
using Random
using Distributions
using LabelledArrays
using OrdinaryDiffEq
using Catlab.CategoricalAlgebra
using Plots

# Subtypes are expected to come with at least a function `interpret(s)` which
# returns an Interpretation
abstract type ModelSemantics end

# The result from running an interpretation of a model Semantics. This is left
# vague so that we have the freedom to implement any imaginable type of semantic
# interpretation
abstract type Interpretation end

# An Interpretation that includes a trajectory of the stock values over time.
# We expect subtypes of SimulationInterpretation to have functions
# `timepoints(s)` which gives the timepoints of the trajectory, and `traj(s)`
# which gives a vector of LVectors representing the states
abstract type SimulationInterpretation <: Interpretation end

# Tell Plots how to visualize a SimulationInterpretation
@recipe function simplot(t::SimulationInterpretation)
    x=timepoints(t), y=traj(t)
end


struct ODEInterpretation <: SimulationInterpretation
    timepoints::Vector{Float64}
    traj::Vector{LArray}
end
traj(o::ODEInterpretation) = o.traj
timepoints(o::ODEInterpretation) = o.timepoints

# Interpretation of a model as a system of ODEs. I.e. the traditional way of
# interpreting a stock & flow model
struct ODESemantics <: ModelSemantics
    model::FullStockFlow
    ode::Function
    model_params::LArray
    stocks_initial::LArray
    start_time::Float64
    stop_time::Float64
    solver_dt::Float64
    solver_abstol::Float64
    solver_alg

    ODESemantics(
        model::FullStockFlow,
        model_params::LArray,
        stocks_initial::LArray,
        start_time::Float64,
        stop_time::Float64;
        solver_dt::Float64 = 0.05,
        solver_abstol::Float64 = 1.0e-8,
        solver_alg = Tsit5()
    ) = new(
        model,
        odefunction(model),
        model_params,
        stocks_initial,
        start_time,
        stop_time,
        solver_dt,
        solver_abstol,
        solver_alg
    )
end

struct TransitionMatrices
    # row represent flows, column represent stocks; and element of 1 of matrix
    # indicates whether there is a connection between the flow and stock; 0
    # indicates no connection
    inflow::Matrix{Int}
    outflow::Matrix{Int}
    TransitionMatrices(p::AbstractStockAndFlowStructure) = begin
        inflow, outflow = zeros(Int,(nf(p),ns(p))), zeros(Int,(nf(p),ns(p)))
        for i in 1:ni(p)
            inflow[subpart(p, i, :ifn), subpart(p, i, :is)] += 1
        end
        for o in 1:no(p)
            outflow[subpart(p, o, :ofn), subpart(p, o, :os)] += 1
        end
        new(inflow, outflow)
    end
end

valueat(x::Number, u, p, t) = x
valueat(f::Number, u, uN, p, t) = x
valueat(f::Function, u, p, t) = f(u, p, t)
valueat(f::Function, u, uN, p, t) = f(u, uN, p, t)


# test argumenterror -- stocks in function of flow "fn" are not linked!
# TODO: find method to generate the exact wrong stocks' names and output in
# error message
function ftest(f::Function, u, p, fn)
    try
        f(u,p,0)
    catch e
        if isa(e, ArgumentError)
            println(
                "Stocks used in the function of flow " *
                "$(fn) are not linked but used!"
            )
            rethrow(e)
        end
    end
end

# if the function f runs to the end, then throw an ErrorException error!
function ferror(f::Function, u, p, fn, umissed)
    f(u, p, 0)
    throw(
        ErrorException(
            "stocks $(umissed) in the function of " *
            "flow $(fn) are linked but not used!"
        )
    )
end

# test stocks in function of flow "fn" are missed!
function fmisstest(f::Function, u, p, fn, umissed)
    try
        ferror(f, u, p, fn, umissed)
    catch e
        if isa(e, ErrorException)
            rethrow(e)
        end
    end
end

function odefunction(sf::FullStockFlow)
    ϕ = funcFlows(sf)
    tm = TransitionMatrices(sf)

    f(du, u, p, t) = begin

        u_m = [u[sname(sf, i)] for i in 1:ns(sf)]
        ϕ_m = [ϕ[fname(sf, i)] for i in 1:nf(sf)]

        for i in 1:ns(sf)
            stockname = sname(sf, i)
            du[stockname] = 0
            for j in 1:nf(sf)
                if tm.inflow[j, i] == 1
                    du[stockname] = du[stockname] + valueat(ϕ_m[j], u, p, t)
                end
                if tm.outflow[j, i] == 1
                    du[stockname] = du[stockname] - valueat(ϕ_m[j], u, p, t)
                end
            end
        end
        return du
    end
    return f
end

# generate an array of all arguments of an expression
function generate_expr_args(expr)
    args = expr.args
    argsarray = []
    for arg in args
        if arg isa Expr
            push!(argsarray, generate_expr_args(arg)...)
        else
            push!(argsarray, arg)
        end
    end
    ops = vcat(collect(values(Operators))...)
    return setdiff(unique(argsarray), ops)
end


""" return the function of the variable at index v """
funcDynam(p::AbstractStockAndFlow, v) = subpart(p, v, :funcDynam)

""" return the function of the variable at index v """
function funcDynam(sf::AbstractStockAndFlowF, v)
    expr = make_v_expr(sf, v)
    args = generate_expr_args(expr)

    args_s = args[findall(in(snames(sf)), args)]
    args_sv = args[findall(in(svnames(sf)), args)]
    args_p = args[findall(in(pnames(sf)), args)]

    generated_func = eval(
        Expr(
            :->,
            Expr(:tuple, args_s..., args_sv..., args_p...),
            Expr(:block, :(()), expr)
        )
    )

    f(u, uN, p, t) = begin
        us = map(i -> u[i], args_s)
        uNs = map(i -> uN[i](u, t), args_sv)
        ps =  map(i -> p[i], args_p)
        return generated_func(us..., uNs..., ps...)
    end
    return f
end

"""
return the function (with no sum variable functions substituted yet)
of the flow at index f
"""
funcFlowRaw(p::FullStockFlow, f) = funcDynam(p, flowVariableIndex(p, f))

"""
return the LVector of pairs: fname => function
(raw: not substitutes the function of sum variables yet)
"""
function funcFlowsRaw(p::FullStockFlow)
    fnames = [fname(p, f) for f in 1:nf(p)]
    return LVector(;[(fnames[f] => funcFlowRaw(p, f)) for f in 1:nf(p)]...)
end

""" generate the function substituting sum variables in with flow index fn """
function funcFlow(pn::FullStockFlow, fn)
    func = funcFlowRaw(pn, fn)
    uN = funcSVs(pn)
    f(u, p, t) = begin
        return valueat(func, u, uN, p, t)
    end
    return f
end

"""
return the LVector of pairs: fname => function
(with function of sum variables substitue in)
"""
function funcFlows(p::FullStockFlow)
    fnames = [fname(p, f) for f in 1:nf(p)]
    return LVector(;[(fnames[f] => funcFlow(p, f)) for f in 1:nf(p)]...)
end

"""
generate the function of a sum auxiliary variable (index sv) with
the sum of all stocks links to it
"""
function funcSV(p::AbstractStockAndFlow0,sv)
    uN(u,t) = begin
        sumS = 0
        for i in stockssv(p,sv)
            sumS=sumS+u[sname(p,i)]
        end
        return sumS
    end
    return uN
end

""" return the LVector of pairs: svname => function """
function funcSVs(p::AbstractStockAndFlow0)
    svnames = [svname(p, sv) for sv in 1:nsv(p)]
    LVector(;[(svnames[sv]=>funcSV(p, sv)) for sv in 1:nsv(p)]...)
end

function interpret(s::ODESemantics)::ODEInterpretation
    prob = ODEProblem(
        s.ode,
        s.stocks_initial,
        (s.start_time, s.stop_time),
        s.model_params
    )
    sol = solve(
        prob,
        s.solver_alg,
        abstol=s.solver_abstol
    )
    timepoints::Vector{Float64} = Vector(s.start_time:s.stop_time)

    return ODEInterpretation(
        timepoints,
        sol(timepoints).u
    )
end



end # Semantics module
