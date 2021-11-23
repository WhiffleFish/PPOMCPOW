using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools

pomdp = LightDark1D()
sol = RootParallelPOWSolver(procs=4, max_time=0.1, tree_queries=100_000)
planner = solve(sol, pomdp)
a, info = action_info(planner, initialstate(pomdp))

@profiler action_info(planner, initialstate(pomdp))

@code_warntype action_info(planner, initialstate(pomdp))

using BenchmarkTools
@benchmark action_info(planner, $(initialstate(pomdp)))
