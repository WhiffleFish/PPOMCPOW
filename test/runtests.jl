using PPOMCPOW
using POMDPs
using POMDPModels, POMDPModelTools

pomdp = LightDark1D()
sol = ParallelPOWSolver(procs=4)
planner = solve(sol, pomdp)
a, info = action_info(planner, initialstate(pomdp))
