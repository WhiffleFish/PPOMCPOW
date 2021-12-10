using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools

pomdp = LightDark1D()
sol = RootParallelPOWSolver(procs = 10, max_time=0.1, tree_queries=100_000)
planner = solve(solver, pomdp)
b0 = initialstate(pomdp)
action_info(par_planner, b0)
