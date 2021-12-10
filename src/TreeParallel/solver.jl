"""
    TreeParallelPOWSolver
Fields:
- `eps::Float64`:
    Rollouts and tree expansion will stop when discount^depth is less than this.
    default: `0.01`

- `max_depth::Int`:
    Rollouts and tree expension will stop when this depth is reached.
    default: `10`

- `criterion::Any`:
    Criterion to decide which action to take at each node. e.g. `MaxUCB(c)`, `MaxQ`, or `MaxTries`
    default: `MaxUCB(1.0)`

- `final_criterion::Any`:
    Criterion for choosing the action to take after the tree is constructed.
    default: `MaxQ()`

- `tree_queries::Int`:
    Number of iterations during each action() call.
    default: `1000`

- `max_time::Float64`:
    Time limit for planning at each steps (seconds).
    default: `Inf`

- `rng::AbstractRNG`:
    Random number generator.
    default: `Xoroshiro128Star()`

- `node_sr_belief_updater::Updater`:
    Updater for state-reward distribution at the nodes.
    default: `POWNodeFilter()`

- `estimate_value::Any`: (rollout policy can be specified by setting this to RolloutEstimator(policy))
    Function, object, or number used to estimate the value at the leaf nodes.
    If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    If this is a number, the value will be set to that number
    default: `TSRandomRolloutSolver()`

- `enable_action_pw::Bool`:
    Controls whether progressive widening is done on actions; if `false`, the entire action space is used.
    default: `true`

- `check_repeat_obs::Bool`:
    Check if an observation was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same observation from being created. If the observation space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`

- `check_repeat_act::Bool`:
    Check if an action was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same action from being created. If the action space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`

- `k_action::Float64`, `alpha_action::Float64`, `k_observation::Float64`, `alpha_observation::Float64`:
        These constants control the double progressive widening. A new observation
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k: `10`, alpha: `0.5`
"""
@with_kw mutable struct TreeParallelPOWSolver{CRIT1,CRIT2,RNG,UPD,EV} <: AbstractPOMCPSolver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    criterion::CRIT1            = MaxUCB(1.0)
    final_criterion::CRIT2      = MaxQ()
    tree_queries::Int           = 1000
    max_time::Float64           = Inf
    rng::RNG                    = Xoroshiro128Star()
    node_sr_belief_updater::UPD = POWNodeFilter()

    estimate_value::EV          = TSRandomRolloutSolver()

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true
    tree_in_info::Bool          = false

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
end

function MCTS.convert_estimator(ev::TSRandomRolloutSolver, solver::TreeParallelPOWSolver, pomdp::POMDP)
    return TSRandomRolloutEstimator(actions(pomdp))
end
