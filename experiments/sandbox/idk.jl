using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools
using POMCPOW

pomdp = LightDark1D()
sol = RootParallelPOWSolver(procs=4, tree_queries=100_000)
planner = solve(sol, pomdp)
a, info = action_info(planner, initialstate(pomdp))

@benchmark action_info(planner, initialstate(pomdp))

@profiler action_info(planner, initialstate(pomdp))

@profiler action_info(planner, initialstate(pomdp))

@code_warntype action_info(planner, initialstate(pomdp))

using BenchmarkTools
@benchmark action_info(planner, $(initialstate(pomdp)))


##
pow_sol = POMCPOWSolver(tree_queries = 10_000)
pow_planner = solve(pow_sol, pomdp)
par_sol = PPOMCPOW.LeafParallelPOWSolver(pow_sol, 4)
par_planner = solve(par_sol, pomdp)
@profiler a, info = action_info(par_planner, initialstate(pomdp))
@profiler a, info = action_info(pow_planner, initialstate(pomdp))

@benchmark action_info(par_planner, initialstate(pomdp))

@benchmark action_info(pow_planner, initialstate(pomdp))

using BenchmarkTools
@benchmark action_info(planner, initialstate(pomdp))

sol = LeafParallelPOWSolver(procs=1, tree_queries=10_000)
planner = solve(sol, pomdp)
@benchmark action_info(planner, initialstate(pomdp))


##
using MCTS, Random, POMDPSimulators, POMDPPolicies
pomdp = LightDark1D()
b0 = initialstate(pomdp)
s0 = rand(b0)

N = 10_000
n_procs = 4
max_steps = 10

est_sol = PPOMCPOW.ParallelRandomRolloutSolver(Random.MersenneTwister(), 5)
est = MCTS.convert_estimator(est_sol, nothing, pomdp)
@benchmark MCTS.estimate_value(est, pomdp, s0, nothing, max_steps)

@benchmark MCTS.estimate_value(est, pomdp, s0, nothing, max_steps)

@benchmark MCTS.estimate_value(est, pomdp, s0, nothing, max_steps)


@benchmark MCTS.estimate_value(est, pomdp, s0, nothing, max_steps)

ro = RolloutSimulator(max_steps=max_steps)
p = RandomPolicy(pomdp)
@benchmark simulate(ro, pomdp, p, updater(p), b0, s0)
