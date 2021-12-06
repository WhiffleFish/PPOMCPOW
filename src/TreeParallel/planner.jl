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
    A = actiontype(pomcp.problem)
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    local iter::Int
    local t0::Float64
    # try
        a, iter, t0 = search(pomcp, tree)
    # catch ex
    #     if ex isa AllSamplesTerminal
    #         @warn("All Samples Terminal")
    #         a = rand(actions(pomcp.problem))
    #     else
    #         throw(ex)
    #     end
    # end
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

    chnl = Channel(5)
    prod_task = Threads.@spawn task_producer(chnl, t0, pomcp)
    n = (Threads.nthreads() ÷ 2) - 1
    #  for i in 1:n
    @sync Threads.@spawn task_taker(chnl, pomcp, tree)
    # end

    iter = fetch(prod_task)

    length(tree.n) == 1 && throw(AllSamplesTerminal(tree.root_belief))

    best_node = select_best(tree, pomcp.solver.final_criterion, TreeParallelPOWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node], iter, t0
end

function task_producer(chnl::Channel, t0::Float64, pomcp::TreeParallelPOWPlanner)
    max_iter = pomcp.solver.tree_queries
    max_time = pomcp.solver.max_time
    iter = 0
    while time() - t0 < max_time && iter < max_iter
        if length(chnl.data) < 2
            @show iter
            iter += 1
            put!(chnl, :job)
        end
    end
    close(chnl)
    return iter
end

function task_taker(chnl::Channel, pomcp::TreeParallelPOWPlanner, tree::TreeParallelPOWTree)
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )

    while isopen(chnl)
        println("Channel Open")
        @show isready(chnl)
        take!(chnl)
        println("Thread $(Threads.threadid()) took a job")
        rng = pomcp.rngs[Threads.threadid()]
        s = rand(rng, tree.root_belief)
        if !isterminal(pomcp.problem, s)
            simulate(pomcp, TreeParallelPOWTreeObsNode(tree, 1), s, max_depth, rng)
        end
        println("Thread $(Threads.threadid()) completed a job")
    end
end
