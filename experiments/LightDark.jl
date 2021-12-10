using QuickPOMDPs
using POMDPModelTools
using Distributions
#=
The simple 1-D light dark problem from https://arxiv.org/pdf/1709.06196v6.pdf, section 5.2, or https://slides.com/zacharysunberg/defense-4#/39
=#

const R = 60
const LIGHT_LOC = 10

const LightDarkPOMDP = QuickPOMDP(
    states = -R:R+1,                  # r+1 is a terminal state
    stateindex = s -> s + R + 1,
    actions = [-10, -1, 0, 1, 10],
    discount = 0.95,
    isterminal = s::Int -> s==R::Int+1,
    obstype = Float64,

    transition = function (s::Int, a::Int)
        if a == 0
            return Deterministic{Int}(R::Int+1)
        else
            return Deterministic{Int}(clamp(s+a, -R::Int, R::Int))
        end
    end,

    observation = (s, a, sp) -> Normal(sp, abs(sp - LIGHT_LOC::Int) + 1e-3),

    reward = function (s, a)
        if a == 0
            return s == 0 ? 100.0 : -100.0
        else
            return -1.0
        end
    end,

    initialstate = POMDPModelTools.Uniform(div(-R::Int,2):div(R::Int,2))
)

POW_params = Dict{Symbol,Any}(
    :criterion => MaxUCB(90.0),
    :k_observation => 5.0,
    :alpha_observation => 1/15,
    :enable_action_pw => false,
    :check_repeat_obs => false,
    :tree_queries => 10_000_000,
    :default_action => (args...) -> rand(actions(pomdp))
)

pomdp = LightDarkPOMDP
max_time = 0.1
sol = POMCPOWSolver(max_time = max_time; POW_params...)
leaf_sol = LeafParallelPOWSolver(sol, 4)
root_sol = RootParallelPOWSolver(procs=10, max_time = max_time; POW_params...)
planner = solve(sol, pomdp)
leaf_planner = solve(leaf_sol, pomdp)
root_planner = solve(root_sol, pomdp)


using POMDPSimulators, ParticleFilters, ProgressMeter

bu = BootstrapFilter(pomdp, 10_000)
ro = RolloutSimulator(max_steps=20)

N = 1000
leaf_r = @showprogress[simulate(ro, pomdp, leaf_planner, bu) for _ in 1:N]
root_r = @showprogress[simulate(ro, pomdp, root_planner, bu) for _ in 1:N]
vanilla_r = @showprogress[simulate(ro, pomdp, planner, bu) for _ in 1:N]

μ_leaf = mean(leaf_r)
μ_root = mean(root_r)
μ_vanilla = mean(vanilla_r)

σ_leaf = std(leaf_r)/sqrt(N)
σ_root = std(root_r)/sqrt(N)
σ_vanilla = std(vanilla_r)/sqrt(N)


##

times = 10.0 .^ (-2:0.25:0)
N = 100

leaf_mean_hist = Float64[]
leaf_stderr_hist = Float64[]

root_mean_hist = Float64[]
root_stderr_hist = Float64[]

vanilla_mean_hist = Float64[]
vanilla_stderr_hist = Float64[]

for t in times
    @show t
    t > 0.1 && (N = 50)

    sol = POMCPOWSolver(max_time = t; POW_params...)
    leaf_sol = LeafParallelPOWSolver(sol, 4)
    root_sol = RootParallelPOWSolver(procs=10, max_time =t; POW_params...)
    planner = solve(sol, pomdp)
    leaf_planner = solve(leaf_sol, pomdp)
    root_planner = solve(root_sol, pomdp)

    leaf_r_t = @showprogress [simulate(ro, pomdp, leaf_planner, bu) for _ in 1:N]
    root_r_t = @showprogress [simulate(ro, pomdp, root_planner, bu) for _ in 1:N]
    vanilla_r_t = @showprogress [simulate(ro, pomdp, planner, bu) for _ in 1:N]

    μ_leaf = mean(leaf_r_t)
    μ_root = mean(root_r_t)
    μ_vanilla = mean(vanilla_r_t)

    σ_leaf = std(leaf_r_t)/sqrt(N)
    σ_root = std(root_r_t)/sqrt(N)
    σ_vanilla = std(vanilla_r_t)/sqrt(N)

    push!(leaf_mean_hist, μ_leaf)
    push!(root_mean_hist, μ_root)
    push!(vanilla_mean_hist, μ_vanilla)

    push!(leaf_stderr_hist, σ_leaf)
    push!(root_stderr_hist, σ_root)
    push!(vanilla_stderr_hist, σ_vanilla)
end

# using CairoMakie
#
# function plot_ax!(f, times, means, stderrs, names::Vector{String}; ci::Number=2, legend=true)
#
#     axis = Axis(
#         f,
#         title = "LightDark Benchmark",
#         xlabel = "Planning Time (sec) - Log Scale",
#         ylabel = "Reward",
#         xscale = log10,
#         xticks = 10. .^ [-2,-1,0],
#         xminorticksvisible = true,
#         xminorgridvisible = true,
#         xminorticks = IntervalsBetween(9),
#         limits = (0.01, 1.0, nothing, nothing)
#     )
#     color_vec = [
#         :blue,
#         :orange,
#         :green,
#         :purple
#     ]
#     line_arr = []
#     t = times
#     for i in eachindex(means)
#         color = color_vec[i]
#         μ = means[i]
#         σ = stderrs[i]
#         l = lines!(t, μ, marker=:rect, color=color, linestyle=:dash)
#         b = band!(t, μ .- ci*σ, μ .+ ci*σ, color=(color,0.5))
#         push!(line_arr, [l,b])
#     end
#     if legend
#         axislegend(
#             axis, line_arr, names,
#             position = :lt, labelsize=10, framevisible=true
#         )
#     end
#     return axis
# end

means = [leaf_mean_hist, root_mean_hist, vanilla_mean_hist]
stderrs = [leaf_stderr_hist, root_stderr_hist, vanilla_stderr_hist]
names = ["Leaf Parallel", "Root Parallel", "Vanilla"]

f = CairoMakie.Figure(resolution = (1000, 700))
axis = CairoMakie.Axis(
    f,
    title = "LightDark Benchmark",
    xlabel = "Planning Time (sec) - Log Scale",
    ylabel = "Reward",
    xscale = log10,
    xticks = 10. .^ [-2,-1,0],
    xminorticksvisible = true,
    xminorgridvisible = true,
    xminorticks = IntervalsBetween(9),
    limits = (0.01, 1.0, nothing, nothing)
)

plot_ax!(f, times, means, stderrs, names)
f

using Plots

Plots.plot(
    times,
    reduce(hcat,means),
    ribbon = reduce(hcat,stderrs),
    xaxis=:log,
    labels = reshape(names,(1,3))
)
Plots.xlabel!("Planning Time")
Plots.ylabel!("Returns")
Plots.title!("LightDark Benchmark")
Plots.savefig("LightDarkComparison.png")


save("LightDarkComparison.png", f)
