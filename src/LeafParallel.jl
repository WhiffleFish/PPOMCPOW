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

struct LeafParallelPOWSolver{POW <: POMCPOWSolver, RNG <: Random.AbstractRNG}
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
