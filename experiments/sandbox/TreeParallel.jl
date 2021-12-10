using Revise
using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools
using JET
using Logging

pomdp = LightDark1D()
sol = TreeParallelPOWSolver(max_time = 1.0, tree_queries=10, enable_action_pw=false)
planner = solve(sol, pomdp)

b0 = initialstate(pomdp)
JET.@report_call action(planner, b0)

@profiler action(planner, b0)

debuglogger = ConsoleLogger(stderr, Logging.Debug)

with_logger(debuglogger) do
    action_info(planner, b0)
end

## debugging
