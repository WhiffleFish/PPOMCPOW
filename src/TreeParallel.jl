#=
Tree Parallelization with Virtual Loss or something...
=#

struct TreeParallelPOWSolver{SOL<:POMCPOWSolver} end
struct TreeParallelPOWPlanner{SOL<:POMCPOWSolver, P<:POMDP} end
