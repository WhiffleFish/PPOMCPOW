#=
Leaf Parallelism
- Instead of rolling out a single particle one time, rollout the particle \
multiple times in parallel.
=#

struct ParallelRandomRolloutSolver{RNG<:Random.AbstractRNG}
    rng::RNG
    procs::Int
end

# Feeding Random.GLOBAL_RNG for parallel rollout estimator rng fails
ParallelRandomRolloutSolver(procs::Int) = ParallelRandomRolloutSolver(Xoroshiro128Star(), procs)

struct ParallelRandomRolloutEstimator{A, RNG<:Random.AbstractRNG}
    rngs::Vector{RNG}
    actions::A
end

"""
Convert `ParallelRandomRolloutSolver` to `ParallelRandomRolloutEstimator`
"""
function convert_estimator(estimator::ParallelRandomRolloutSolver, ::Any, pomdp::POMDP)
    rng_vec = [
        Random.seed!(deepcopy(estimator.rng), rand(UInt64)) for _ in 1:estimator.procs
        ]
    return ParallelRandomRolloutEstimator(rng_vec, actions(pomdp))
end

# estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
function estimate_value(estimator::ParallelRandomRolloutEstimator, pomdp::POMDP{S}, s::S, ::Any, depth::Int) where S
    v_sum = Folds.mapreduce(rng -> rollout(estimator, rng, pomdp, s, depth), +, estimator.rngs)
    return v_sum/length(estimator.rngs)
end

function rollout(estim::ParallelRandomRolloutEstimator, rng::Random.AbstractRNG, pomdp::POMDP{S}, s::S, depth::Int) where {S}

    disc = 1.0
    r_total = 0.0
    step = 1

    while !isterminal(pomdp, s) && step ≤ depth

        a = rand(rng, estim.actions)

        sp,r = @gen(:sp,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end

struct LeafParallelPOWSolver{POW <: POMCPOWSolver} <: Solver
    solver::POW
    procs::Int
end

# Feeding Random.GLOBAL_RNG for parallel rollout estimator rng fails
function LeafParallelPOWSolver(;procs::Int=1, kwargs...)
    return POMCPOWSolver(;
        estimate_value = ParallelRandomRolloutSolver(procs),
        kwargs...
    )
end

struct LeafParallelPOWPlanner{POW<:POMCPOWPlanner} <: Policy
    powplanner::POW
    procs::Int
end

function POMDPs.solve(sol::LeafParallelPOWSolver, pomdp::POMDP)
    pomcp = sol.solver
    pomcp.estimate_value = ParallelRandomRolloutSolver(sol.procs)
    return LeafParallelPOWPlanner(solve(pomcp, pomdp), sol.procs)
end

function POMDPModelTools.action_info(planner::LeafParallelPOWPlanner, b)
    t0 = time()
    pomcp = planner.powplanner
    A = actiontype(pomcp.problem)
    info = Dict{Symbol, Any}()
    tree = POMCPOW.make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        a = new_search(pomcp, tree, info)
        if pomcp.solver.tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, POMCPOW.default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

function new_search(pomcp::POMCPOWPlanner, tree::POMCPOWTree, info::Dict{Symbol,Any}=Dict{Symbol,Any}())
    all_terminal = true
    i = 0
    max_depth = min(
        pomcp.solver.max_depth,
        ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem)))
    )
    t0 = time()
    while i < pomcp.solver.tree_queries
        i += 1
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)

            POMCPOW.simulate(pomcp, POWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end
        if time() - t0 > pomcp.solver.max_time
            break
        end
    end
    info[:search_time] = time() - t0
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = POMCPOW.select_best(pomcp.solver.final_criterion, POWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end

POMDPs.action(planner::LeafParallelPOWPlanner, b) = first(action_info(planner, b))
