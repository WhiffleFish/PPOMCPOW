"""
Thread-safe random rollout solver
- takes rng argument for value estimation to prevent multiple threads from
overwriting the same RNG state
"""
struct TSRandomRolloutSolver end

struct TSRandomRolloutEstimator{A}
    actions::A
end

action(rng::AbstractRNG, ve::TSRandomRolloutEstimator) = rand(rng, ve.actions)

function MCTS.estimate_value(est::TSRandomRolloutEstimator, pomdp::POMDP{S}, s::S, d::Int, rng::AbstractRNG) where S
    disc = 1.0
    r_total = 0.0
    step = 1
    γ = discount(pomdp)

    while !isterminal(pomdp, s) && step ≤ d

        a = action(rng, est)

        sp,r = @gen(:sp,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        disc *= γ
        step += 1
    end

    return r_total
end
