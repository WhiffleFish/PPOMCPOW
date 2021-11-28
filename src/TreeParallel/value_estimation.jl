struct TSRandomRolloutSolver end

struct TSRandomRolloutEstimator{A}
    actions::A
end

action(rng::AbstracRNG, ve::TSRandomRolloutEstimate) = rand(rng, ve.actions)

function MCTS.convert_estimator(ev::TSRandomRolloutSolver, solver::TreeParallelPOWSolver, pomdp::POMDP)
    return TSRandomRolloutEstimator(actions(pomdp))
end

function MCTS.estimate_value(est::TSRandomRolloutEstimator, pomdp::POMDP{S}, s::S, d::Int, rng::AbstractRNG)
    disc = 1.0
    r_total = 0.0
    step = 1
    γ = discount(pomdp)

    while !isterminal(pomdp, s) && step ≤ depth

        a = action(rng, est)

        sp,r = @gen(:sp,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        disc *= γ
        step += 1
    end

    return r_total
end
