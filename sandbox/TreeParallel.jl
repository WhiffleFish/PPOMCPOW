using Revise
using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools
using JET
# using JETTest

pomdp = LightDark1D()
sol = TreeParallelPOWSolver(max_time = 1.0, tree_queries=10_000, enable_action_pw=false)
planner = solve(sol, pomdp)

b0 = initialstate(pomdp)
JET.@report_call action(planner, b0)
# JETTest.@report_dispatch action(planner, b0)

@profiler action(planner, b0)

action_info(planner, b0)
