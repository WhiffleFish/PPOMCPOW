using PPOMCPOW
using POMDPs
using POMDPModels, POMDPModelTools

pomdp = LightDark1D()
sol = ParallelPOWSolver(procs=4, max_time=Inf, tree_queries=10_000)
planner = solve(sol, pomdp)
a, info = action_info(planner, initialstate(pomdp))

@profiler action_info(planner, initialstate(pomdp))

@code_warntype action_info(planner, initialstate(pomdp))

planner.planners[1].solver.tree_queries
