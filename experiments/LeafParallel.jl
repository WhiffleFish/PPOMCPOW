using PPOMCPOW
using POMCPOW
using POMDPs, POMDPModels, POMDPModelTools, POMDPSimulators

pomdp = TigerPOMDP()
sol = POMCPOWSolver(max_time = 0.1, tree_queries=100_000)
leaf_sol = LeafParallelPOWSolver(sol, 4)
root_sol = RootParallelPOWSolver(procs=10, max_time = 0.1, tree_queries=100_000)
planner = solve(sol, pomdp)
leaf_planner = solve(leaf_sol,pomdp)
root_planner = solve(root_sol,pomdp)
b0 = initialstate(pomdp)
action(root_planner, b0)
action(leaf_planner, b0)
action(planner, b0)

## When do we not listen?
using Statistics, ProgressMeter
N = 1000
leaf_actions = @showprogress [action(leaf_planner, b0) for _ in 1:N]
root_actions = @showprogress [action(root_planner, b0) for _ in 1:N]
vanilla_actions = @showprogress [action(planner, b0) for _ in 1:N]

μ_leaf = sum(leaf_actions .== 0)/N
μ_root = sum(root_actions .== 0)/N
μ_vanilla = sum(vanilla_actions .== 0)/N

std_err_leaf = std(leaf_actions .== 0) / √N
std_err_root = std(root_actions .== 0) / √N
std_err_vanilla = std(vanilla_actions .== 0) / √N

using Plots
plt = bar(
    ["Leaf Parallel", "Root Parallel", "Vanilla"],
    [μ_leaf, μ_root, μ_vanilla],
    yerror = [std_err_leaf, std_err_root, std_err_vanilla],
    label = "",
    ylabel= "Listen Action Proportion",
    title = "Tiger Initial Action")

savefig(plt, "TigerInitialActionFullCompare.png")

##

ro = RolloutSimulator(max_steps = 20)
simulate(ro, pomdp, par_planner, updater(planner))
