 # previously `search`
function build_tree!(pomcp::POMCPOWPlanner, tree)
    iter = 0
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    t0 = time()
    while iter < pomcp.solver.tree_queries && time() - t0 < pomcp.solver.max_time
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            iter += 1
            POMCPOW.simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth)
        end
    end

    return iter == 0
end

# previouly `action_info`
function tree_info(pomcp::POMCPOWPlanner{P,NBU}, b) where {P,NBU}
    #= TODO:
    make_tree rebuilds the whole tree from scratch at each action call
    kicking up a lot of GC. Consider `empty!` as a non_allocating alternative.
    =#
    tree = POMCPOW.make_tree(pomcp, b)
    pomcp.tree = tree
    empty_tree = build_tree!(pomcp, tree)
    return tree, empty_tree
end

struct RootParallelPOWSolver{SOL<:POMCPOWSolver}
    powsolver::SOL
    procs::Int
end

#=
    How do we ensure that all baseQ calls run concurrently in `action_info` ?
        default to `procs = Threads.nthreads() ÷ 2` ?
        `procs = Distributed.nprocs()` then `pmap` ?
=#
function RootParallelPOWSolver(;procs::Int=1, kwargs...)
    return RootParallelPOWSolver(
        POMCPOWSolver(;kwargs...),
        procs
    )
end

struct RootParallelPOWPlanner{P<:POMDP, POW<:POMCPOWPlanner}
    pomdp::P
    planners::Vector{POW}
end

function RootParallelPOWPlanner(pomdp::POMDP, planner::POMCPOWPlanner, n::Int)
    planner_vec = Vector{POMCPOWPlanner}(undef, n)
    for i in 1:n
        p = deepcopy(planner)
        Random.seed!(p.solver.rng, rand(UInt32))
        planner_vec[i] = p
    end
    return RootParallelPOWPlanner(pomdp,planner_vec)
end

function POMDPs.solve(solver::RootParallelPOWSolver, pomdp::POMDP)
    return RootParallelPOWPlanner(pomdp, solve(solver.powsolver, pomdp), solver.procs)
end

function baseQ(planner::POMCPOWPlanner{P}, b) where P
    tree, empty_tree = tree_info(planner, b)

    root = first(tree.tried)
    A = actiontype(P)
    Q_vec = Vector{Pair{A,Float64}}(undef, length(root))

    for (i,anode) in enumerate(root)
        a = tree.a_labels[anode]
        v = tree.v[anode]
        Q_vec[i] = (a => v)
    end

    return Q_vec
end

function merge_Q_vecs(all_Qs::Vector{Vector{Pair{A, Float64}}}) where A
    #=
    TODO: Weight Q estimates by number of visits (i.e. the N in UCB eq.)
    Currently performing summation over Q which is proportional to
    unweighted average (∑Q ∝ ∑Q/N) so the end effect is the same in terms of
    decision making. However to the end user in `action_info` the inflated
    Q value estimates would look weird because they're not actually
    Q values but rather a *sum* of estimated Q values.
    =#
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
    return a_opt::A
end

function POMDPModelTools.action_info(planner::RootParallelPOWPlanner, b)
    t0 = time()
    planner_vec = planner.planners
    A = actiontype(planner.pomdp)
    vec = Vector{Vector{Pair{A, Float64}}}(undef, length(planner_vec))

    Threads.@threads for i in eachindex(vec)
        vec[i] = baseQ(planner_vec[i], b)
    end

    Q_dict = merge_Q_vecs(vec)
    a_opt = maximizing_action(Q_dict)
    return a_opt, (time=time()-t0,Q=Q_dict)
end

POMDPs.action(planner::RootParallelPOWPlanner, b) = first(action_info(planner, b))
