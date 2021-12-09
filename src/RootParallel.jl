#=
Root Parallelism
- Build multiple trees independently
- Combine root Q-values of trees to choose action
=#

"""
Modification of `search` in POMCPOW.jl

Constructs a POMCPOW tree according to planner parameters and returns
boolean that's true if all sampled particles in the root belief are terminal
indicating that the tree does not extend past the root belief.
"""
function build_tree!(pomcp::POMCPOWPlanner, tree)
    iter = 0
    all_terminal = true
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    t0 = time()
    while iter < pomcp.solver.tree_queries && time() - t0 < pomcp.solver.max_time
        s = rand(pomcp.solver.rng, tree.root_belief)
        iter += 1
        if !POMDPs.isterminal(pomcp.problem, s)
            POMCPOW.simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end
    end

    return all_terminal
end


"""
Modification of `action_info` in POMCPOW.jl

Initializes a POMCPOW tree with `make_tree` according to root belief `b`, then
constructs tree according to `POMCPOWPlanner` parameters. Finally, returns
constructed tree and boolean indicating whether or not the constructed tree is
empty (either due to root belief being terminal or not enough time alotted to
complete even a single tree query).
"""
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

struct RootParallelPOWSolver{SOL<:POMCPOWSolver} <: Solver
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

struct RootParallelPOWPlanner{P<:POMDP, POW<:POMCPOWPlanner} <: Policy
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


"""
Given a POMCPOW tree, return vector containing pairs `a => Q(b,a)` where `b` is
the root belief of the tree and `a` is an action explored by the planner at the
root.
"""
function baseQ(planner::POMCPOWPlanner{P}, b) where P
    tree, empty_tree = tree_info(planner, b)

    root = first(tree.tried)
    A = actiontype(P)
    Q_vec = Vector{Tuple{A,Float64,Int}}(undef, length(root))

    for (i,anode) in enumerate(root)
        a = tree.a_labels[anode]
        v = tree.v[anode]
        n = tree.n[anode]
        Q_vec[i] = (a,v,n)
    end

    return Q_vec
end


"""
Merge Q values produced by multiple trees

INPUT: Vector of outputs produced by `baseQ`
OUTPUT: Single dictionary mapping `a => Q(b,a)`
"""
function merge_Q_vecs(all_Qs::Vector{Vector{Tuple{A, Float64, Int}}}) where A
    #=
    TODO: Weight Q estimates by number of visits (i.e. the N in UCB eq.)
    Currently performing summation over Q which is proportional to
    unweighted average (∑Q ∝ ∑Q/N) so the end effect is the same in terms of
    decision making. However to the end user in `action_info` the inflated
    Q value estimates would look weird because they're not actually
    Q values but rather a *sum* of estimated Q values.
    =#
    count_dict = Dict{A,Int}()
    final_dict = Dict{A, Float64}()
    for planner_Qs in all_Qs
        for (a,v,n) in planner_Qs
            q = get(final_dict, a, 0.0)
            N = get(count_dict, a, 0)
            n_tot = n + N
            Q = q + (1/n_tot)*(v - q)
            final_dict[a] = Q
            N == 0 ? (count_dict[a] = n) : (count_dict[a] = n)
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
    vec = Vector{Vector{Tuple{A,Float64,Int}}}(undef, length(planner_vec))

    Threads.@threads for i in eachindex(vec)
        vec[i] = baseQ(planner_vec[i], b)
    end

    Q_dict = merge_Q_vecs(vec)
    a_opt = maximizing_action(Q_dict)
    return a_opt, (time=time()-t0,Q=Q_dict)
end

POMDPs.action(planner::RootParallelPOWPlanner, b) = first(action_info(planner, b))
