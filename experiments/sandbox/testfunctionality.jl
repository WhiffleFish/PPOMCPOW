using PPOMCPOW
using POMDPs, POMDPModels

pomdp = LightDark1D()
sol = TreeParallelPOWSolver(max_time = Inf, tree_queries=10, enable_action_pw=false)
planner = solve(sol, pomdp)
b0 = initialstate(pomdp)
