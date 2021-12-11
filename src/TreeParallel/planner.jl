mutable struct TreeParallelPOWPlanner{SOL,P,NBU,C,SE,RNG<:AbstractRNG} <: Policy
    solver::SOL
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    solved_estimate::SE
    tree::Union{Nothing,TreeParallelPOWTree} # BAD TODO: FIX
    rngs::Vector{RNG}
end

function TreeParallelPOWPlanner(solver::TreeParallelPOWSolver, problem::POMDP)
    rngs = [
        Random.seed!(deepcopy(solver.rng), rand(UInt64))
        for _ in 1:nthreads()
    ]
    return TreeParallelPOWPlanner(
        solver,
        problem,
        solver.node_sr_belief_updater,
        solver.criterion,
        convert_estimator(solver.estimate_value, solver, problem),
        nothing,
        rngs
    )
end

function POMDPs.solve(solver::TreeParallelPOWSolver, problem::POMDP)
    return TreeParallelPOWPlanner(solver, problem)
end

Random.seed!(p::TreeParallelPOWPlanner, seed) = Random.seed!(p.solver.rng, seed)

function POMDPModelTools.action_info(pomcp::TreeParallelPOWPlanner, b)
    t0 = time()
    A = actiontype(pomcp.problem)
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    local iter::Int
    local t0::Float64
    try
        a, iter = search(pomcp, tree)
    catch ex
        if ex isa AllSamplesTerminal
            @warn("All Samples Terminal")
            a = rand(actions(pomcp.problem))
            iter = 0
        else
            # throw(ex)
            @warn typeof(ex)
            a = rand(actions(pomcp.problem))
            iter = 0
        end
    end
    return a, (tree_queries=iter, time=time()-t0)
end

POMDPs.action(pomcp::TreeParallelPOWPlanner, b) = first(action_info(pomcp, b))

function make_tree(p::TreeParallelPOWPlanner{SOL,P,NBU}, b) where {SOL,P,NBU}
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    B = belief_type(NBU,P)
    return TreeParallelPOWTree{B, A, O, typeof(b)}(b, 2*min(100_000, p.solver.tree_queries))
end

function search(pomcp::TreeParallelPOWPlanner, tree::TreeParallelPOWTree)
    t0 = time()
    max_iter = pomcp.solver.tree_queries
    max_time = pomcp.solver.max_time
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    rngs = pomcp.rngs
    ##

    # println("Before sim start")
    # methodology from  https://github.com/kykim0/MCTS.jl/blob/dev/src/vanilla.jl
    timeout = t0 + max_time
    sim_channel = Channel{Task}(min(1000, max_iter), spawn=true) do channel
        for n in 1:max_iter
            put!(channel, Threads.@spawn begin
                rng = rngs[threadid()]
                simulate(
                    pomcp,
                    TreeParallelPOWTreeObsNode(tree, 1),
                    rand(rng, tree.root_belief), max_depth, rng)
                end)
        end
        # println("Channel task finished ", channel)
    end

    iter = 0
    for sim_task in sim_channel
        iter += 1
        time() > timeout && break
        try
            # println("Fetch ", counter)
            fetch(sim_task)  # Throws a TaskFailedException if failed.
        catch err
            throw(err.task.exception)  # Throw the underlying exception.
        end
    end

    ##

    length(tree.n) == 1 && throw(AllSamplesTerminal(tree.root_belief))

    best_node = select_best(tree, pomcp.solver.final_criterion, TreeParallelPOWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node], iter
end
