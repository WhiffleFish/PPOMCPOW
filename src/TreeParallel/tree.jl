struct TreeParallelPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Threads.Atomic{Int}}
    v::Vector{Float64}
    generated::Vector{Vector{Pair{O,Int}}}
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A}
    n_a_children::Vector{Threads.Atomic{Int}}

    # observation nodes
    sr_beliefs::Vector{B} # first element is #undef
    total_n::Vector{Threads.Atomic{Int}}
    tried::Vector{Vector{Int}}
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params
    o_labels::Vector{O}

    tree_lock::ReentrantLock
    b_locks::Vector{ReentrantLock} # belief thread locks
    a_locks::Vector{ReentrantLock} # action thread locks

    # root
    root_belief::RB

    function TreeParallelPOWTree{B,A,O,RB}(root_belief, sz::Int=1000) where{B,A,O,RB}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Threads.Atomic{Int}[], sz),
            sizehint!(Threads.Atomic{Int}[], sz),
            sizehint!(Vector{Pair{O,Int}}[], sz),
            Dict{Tuple{Int,O}, Int}(),
            sizehint!(A[], sz),
            sizehint!(Threads.Atomic{Int}[], sz),

            sizehint!(Array{B}(undef, 1), sz),
            sizehint!(Threads.Atomic{Int}[Threads.Atomic{Int}(0)], sz),
            sizehint!(Vector{Int}[Int[]], sz),
            Dict{Tuple{Int,A}, Int}(),
            sizehint!(Array{O}(undef, 1), sz),

            ReentrantLock(),
            sizehint!(Array{ReentrantLock}(undef, 1), sz),
            sizehint!(Array{ReentrantLock}(undef, 1), sz),

            root_belief
        )
    end
end

@inline function push_bnode!(
        pomcp::TreeParallelPOWPlanner,
        tree::TreeParallelPOWTree{B,A,O},
        best_node::Int,
        s, a::A, sp, o::O, r::Real,
        check_repeat_obs=true) where {B,A,O}

    lock.(tree.b_locks)
        hao = length(tree.sr_beliefs) + 1
        push!(tree.sr_beliefs,
              init_node_sr_belief(pomcp.node_sr_belief_updater,
                                  pomcp.problem, s, a, sp, o, r))
        push!(tree.total_n, Threads.Atomic{Int}(0))
        push!(tree.tried, Int[])
        push!(tree.o_labels, o)
        check_repeat_obs && tree.a_child_lookup[(best_node, o)] = hao
    unlock.(tree.b_locks)

    Threads.atomic_add!(tree.n_a_children[best_node], 1)

    nothing
end

@inline function push_anode!(
        tree::TreeParallelPOWTree{B,A,O},
        h::Int,
        a::A,
        n::Int=0,
        v::Float64=0.0,
        update_lookup=true) where {B,A,O}

    lock.(tree.a_locks)
        anode = length(tree.n) + 1
        push!(tree.a_locks, ReentrantLock())
        push!(tree.n, Threads.Atomic{Int}(n))
        push!(tree.v, v)
        push!(tree.generated, Pair{O,Int}[])
        push!(tree.a_labels, a)
        push!(tree.n_a_children, Threads.Atomic{Int}(0))
    unlock.(tree.a_locks)

    lock.(tree.b_locks)
        update_lookup && tree.o_child_lookup[(h, a)] = anode
        push!(tree.tried[h], anode)
    unlock.(tree.b_locks)
    Threads.atomic_add!(tree.total_n[h], n)

    nothing
end

struct TreeParallelPOWTreeObsNode{B,A,O,RB} <: BeliefNode
    tree::TreeParallelPOWTree{B,A,O,RB}
    node::Int
end

isroot(h::TreeParallelPOWTreeObsNode) = h.node==1

@inline function belief(h::TreeParallelPOWTreeObsNode)
    if isroot(h)
        return h.tree.root_belief
    else
        return StateBelief(h.tree.sr_beliefs[h.node])
    end
end

function sr_belief(h::TreeParallelPOWTreeObsNode)
    if isroot(h)
        error("Tried to access the sr_belief for the root node in a POMCPOW tree")
    else
        return h.tree.sr_beliefs[h.node]
    end
end

n_children(h::TreeParallelPOWTreeObsNode) = length(h.tree.tried[h.node])
