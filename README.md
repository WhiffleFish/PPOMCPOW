# Parallel POMCPOW

Offers 3 types of parallelism:
- Root Parallel
- Leaf Parallel
- Tree Parallel

All solvers support all keyword arguments supported by [POMCPOW.jl](https://github.com/JuliaPOMDP/POMCPOW.jl)

## Root Parallel POMCPOW

Constructs multiple trees in parallel and uses weighted average of Q values of all trees to determine value-maximizing action.

The number of trees constructed in parallel is controlled by the extra `procs` kwarg.

```julia
using PPOMCPOW
using POMDPs, POMDPModels

pomdp = LightDark1D()
solver = RootParallelPOWSolver(procs=4, max_time=0.1, tree_queries=100_000)
planner = solve(solver, pomdp)
a = action(solver, initialstate(pomdp))
```

## Leaf Parallel POMCPOW

Evaluates leaf nodes with multithreaded rollouts leading to a higher accuracy and lower variance value estimate.

The number of multithreaded leaf node rollouts is controlled by the `procs` kwarg.

```julia
using PPOMCPOW
using POMDPs, POMDPModels

pomdp = LightDark1D()
solver = LeafParallelPOWSolver(procs=4, max_time=0.1, tree_queries=100_000)
planner = solve(solver, pomdp)
a = action(solver, initialstate(pomdp))
```


## Tree Parallel POMCPOW

Multiple workers traverse the tree simultaneously using [virtual loss](https://liacs.leidenuniv.nl/~plaata1/papers/paper_ICAART17.pdf) to guide the search. The number of traversing workers is automatically determined with `Threads.nthreads()`

```julia
using PPOMCPOW
using POMDPs, POMDPModels

pomdp = LightDark1D()
solver = TreeParallelPOWSolver(max_time=0.1, tree_queries=100_000, max_depth=20)
planner = solve(solver, pomdp)
a = action(solver, initialstate(pomdp))
```
