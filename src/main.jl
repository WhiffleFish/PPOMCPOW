struct ParallelPOWSolver{SOL<:POMCPOWSolver}
    powsolver::SOL
    procs::Int
end

function ParallelPOWSolver(;procs::Int=1, kwargs...)
    return ParallelPOWSolver(
        POMCPOWSolver(kwargs..., tree_in_info=true),
        procs
    )
end

struct ParallelPOWPlanner{POW <: POMCPOWPlanner}
    planners::Vector{POW}
end

function ParallelPOWPlanner(planner::POMCPOWPlanner, n::Int)
    planner_vec = Vector{POMCPOWPlanner}(undef, n)
    for i in 1:n
        p = deepcopy(planner)
        Random.seed!(p.solver.rng, rand(UInt32))
        planner_vec[i] = p
    end
    return ParallelPOWPlanner(planner_vec)
end

function POMDPs.solve(solver::ParallelPOWSolver, pomdp::POMDP)
    return ParallelPOWPlanner(solve(solver.powsolver, pomdp), solver.procs)
end

function baseQ(planner::POMCPOWPlanner{P}, b) where P
    a, info = action_info(planner, b)
    tree = info[:tree]

    root = first(tree.tried)
    Q_vec = Vector{Pair{actiontype(P),Float64}}(undef, length(root))

    for (i,anode) in enumerate(root)
        a = tree.a_labels[anode]
        v = tree.v[anode]
        Q_vec[i] = (a => v)
    end

    return Q_vec
end

function merge_Q_vecs(all_Qs::Vector{Vector{Pair{A, Float64}}}) where A
    final_dict = Dict{A, Float64}()
    for planner_Qs in all_Qs
        for (a,v) in planner_Qs
            q = get(final_dict, a, 0.0)
            final_dict[a] = q + v
        end
    end
    return final_dict
end

function maximizing_action(Q_dict::Dict{A, Float64}) where A
    max_Q = -Inf
    a_opt = nothing
    for (a,Q) in Q_dict
        if Q > max_Q
            max_Q = Q
            a_opt = a
        end
    end
    return a_opt
end

function POMDPModelTools.action_info(planner::ParallelPOWPlanner, b)
    t0 = time()
    planner_vec = planner.planners
    A = actiontype(first(planner_vec).problem)
    vec = Vector{Vector{Pair{A, Float64}}}(undef, length(planner_vec))

    Threads.@threads for i in eachindex(vec)
        vec[i] = baseQ(planner_vec[i], b)
    end

    Q_dict = merge_Q_vecs(vec)
    a_opt = maximizing_action(Q_dict)
    return a_opt, (time=time()-t0,Q=Q_dict)
end

POMDPs.action(planner::ParallelPOWPlanner, b) = first(action_info(planner, b))
