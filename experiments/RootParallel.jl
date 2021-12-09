using PPOMCPOW
using POMCPOW
using POMDPs, POMDPModels, POMDPModelTools, POMDPSimulators

pomdp = TigerPOMDP()
par_sol = RootParallelPOWSolver(procs = 10, max_time=0.1, tree_queries=100_000)
sol = POMCPOWSolver(max_time=0.1, tree_queries=100_000)
planner = solve(sol, pomdp)
par_planner = solve(par_sol,pomdp)
b0 = initialstate(pomdp)
action(par_planner, b0)

## When do we not listen?
using Statistics, ProgressMeter
N = 1000
par_actions = @showprogress [action(par_planner, b0) for _ in 1:N]
vanilla_actions = @showprogress [action(planner, b0) for _ in 1:N]

μ_p = sum(par_actions .== 0)/N
μ_v = sum(vanilla_actions .== 0)/N

std_err_p = std(par_actions .== 0) / √N
std_err_v = std(vanilla_actions .== 0) / √N

using Plots
plt = bar(
    ["Parallel", "Vanilla"],
    [sum(par_actions .== 0)/N, sum(vanilla_actions .== 0)/N],
    yerror = [std_err_p, std_err_v],
    label = "",
    ylabel= "Listen Action Proportion",
    title = "Tiger Initial Action")

savefig(plt, "TigerInitialAction.png")

##

ro = RolloutSimulator(max_steps = 20)
simulate(ro, pomdp, par_planner, updater(planner))
