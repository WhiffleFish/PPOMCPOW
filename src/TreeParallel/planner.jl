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
    info = Dict{Symbol, Any}()
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        a = search(pomcp, tree)
        if pomcp.solver.tree_in_info
            info[:tree] = tree
        end
    catch ex
        if ex isa AllSamplesTerminal
            a = rand(actions(pomcp.problem))
        else
            throw(ex)
        end
    end
    return a, info
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
    iter = Atomic{Int}(0)
    all_terminal = true
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    t0 = time()
    while iter[] < pomcp.solver.tree_queries && time() - t0 < pomcp.solver.max_time
        Threads.@spawn begin
            atomic_add!(iter, 1)
            rng = pomcp.rngs[Threads.threadid()]
            s = rand(rng, tree.root_belief)
            if !POMDPs.isterminal(pomcp.problem, s)
                simulate(pomcp, TreeParallelPOWTreeObsNode(tree, 1), s, max_depth, rng)
                all_terminal = false
            end
        end
    end

    all_terminal && throw(AllSamplesTerminal(tree.root_belief))

    best_node = select_best(tree, pomcp.solver.final_criterion, TreeParallelPOWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end
