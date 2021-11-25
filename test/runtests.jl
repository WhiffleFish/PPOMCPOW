using PPOMCPOW
using POMDPs, POMDPModels, POMDPModelTools, POMDPSimulators, POMDPPolicies
using MCTS
using Test
using Random

@testset "Basic Functionality" begin
    pomdp = LightDark1D()
    sol = RootParallelPOWSolver(procs=4, max_time=0.1, tree_queries=100_000)
    planner = solve(sol, pomdp)
    @inferred action_info(planner, initialstate(pomdp))

    sol = LeafParallelPOWSolver(procs=4, max_time=0.1, tree_queries=100_000)
    planner = solve(sol, pomdp)
    @inferred action_info(planner, initialstate(pomdp))
end

@testset "Leaf Evaluation" begin
    pomdp = LightDark1D()
    b0 = initialstate(pomdp)
    s0 = rand(b0)

    N = 10_000
    n_procs = 4
    max_steps = 10

    est_sol = PPOMCPOW.ParallelRandomRolloutSolver(Random.MersenneTwister(), n_procs)
    est = MCTS.convert_estimator(est_sol, :blank, pomdp)
    N_parallel = N ÷ n_procs
    VE_parallel = sum(
        MCTS.estimate_value(est, pomdp, s0, :blank, max_steps)
        for _ in 1:N_parallel
        ) / N_parallel

    ro = RolloutSimulator(max_steps=max_steps)
    p = RandomPolicy(pomdp)
    VE_sequential = sum(simulate(ro, pomdp, p, updater(p), b0, s0) for _ in 1:N)/N

    @test ≈(VE_parallel, VE_sequential, atol=0.5)
end
