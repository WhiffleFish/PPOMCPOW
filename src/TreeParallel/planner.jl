struct TreeParallelPOWPlanner{P,NBU,C,NA,SE,IN,IV,SOL,TREE<:TreeParallelPOWTree,RNG<:AbstractRNG} <: Policy
    solver::SOL
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    solved_estimate::SE
    init_N::IN
    init_V::IV
    tree::TREE
    rngs::Vector{RNG}
end

function TreeParallelPOWPlanner(solver::TreeParallelPOWSolver, problem::POMDP)
    rngs = [seed!(deepcopy(solver.rng),rand(UInt64)) for _ in 1:Threads.nthreads()]
    POMCPOWPlanner(solver,
                  problem,
                  solver.node_sr_belief_updater,
                  solver.criterion,
                  convert_estimator(solver.estimate_value, solver, problem),
                  solver.init_N,
                  solver.init_V,
                  nothing,
                  rngs
    )
end

function POMDPs.solve(solver::TreeParallelPOWSolver, problem::POMDP)
    return TreeParallelPOWPlanner(solver, problem)
end

Random.seed!(p::TreeParallelPOWPlanner, seed) = Random.seed!(p.solver.rng, seed)

function POMDPModelTools.action_info(pomcp::TreeParallelPOWPlanner{P}, b) where P
    A = actiontype(P)
    info = Dict{Symbol, Any}()
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        a = search(pomcp, tree, info)
        if pomcp.solver.tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

action(pomcp::TreeParallelPOWPlanner, b) = first(action_info(pomcp, b))

function POMDPPolicies.actionvalues(p::TreeParallelPOWPlanner, b)
    tree = make_tree(p, b)
    search(p, tree)
    values = Vector{Union{Float64,Missing}}(missing, length(actions(p.problem)))
    for anode in tree.tried[1]
        a = tree.a_labels[anode]
        values[actionindex(p.problem, a)] = tree.v[anode]
    end
    return values
end

function make_tree(p::TreeParallelPOWPlanner{P, NBU}, b) where {P, NBU}
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    B = belief_type(NBU,P)
    return TreeParallelPOWTree{B, A, O, typeof(b)}(b, 2*min(100_000, p.solver.tree_queries))
end

function search(pomcp::TreeParallelPOWPlanner, tree::TreeParPOMCPOWTree)
    iter = Threads.Atomic{Int}(0)
    all_terminal = true
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    t0 = time()
    while iter < pomcp.solver.tree_queries && time() - t0 < pomcp.solver.max_time
        @spawnat :any begin
            rng = pomcp.rngs[Threads.threadid()]
            s = rand(rng, tree.root_belief)
            Threads.atomic_add!(iter, 1)
            if !POMDPs.isterminal(pomcp.problem, s)
                simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth, rng)
                all_terminal = false
            end
        end
    end

    all_terminal && throw(AllSamplesTerminal(tree.root_belief))

    best_node = select_best(tree, pomcp.solver.final_criterion, POWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end
