struct TreeParallelPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Atomic{Int}} # ba_idx => number of times tried
    v::Vector{Float64} # ba_idx => value of ba_idx
    generated::Vector{Vector{Pair{O,Int}}} # ba_idx => [(o_1=>bao_1), ..., (o_n=>bao_n)]
    a_child_lookup::Dict{Tuple{Int,O}, Int} # (ba_idx,o) => b_idx
    a_labels::Vector{A} # ba_idx => a
    n_a_children::Vector{Atomic{Int}} # ba_idx => n_children

    # observation nodes
    sr_beliefs::Vector{B} # b_idx => particle_belief    (first element is #undef)
    total_n::Vector{Atomic{Int}} # b_idx => num of times b_idx was visited in search
    tried::Vector{Vector{Int}} # b_idx => [ba_idx1, ba_idx2, ..., ba_idxn] (may not even need this if we have n_tried)
    n_tried::Vector{Atomic{Int}} # b_idx => length(tried[b_idx])
    o_child_lookup::Dict{Tuple{Int,A}, Int} # (b_idx,a) => ba_idx
    o_labels::Vector{O} # b'_idx => o (b'=τ(b,a,o)) (first element is #undef)

    tree_lock::ReentrantLock # Lock full tree to prevent any modification by other threads
    b_locks::Vector{ReentrantLock} # belief thread locks
    a_locks::Vector{ReentrantLock} # action thread locks

    # root
    root_belief::RB

    function TreeParallelPOWTree{B,A,O,RB}(root_belief, sz::Int=1000) where{B,A,O,RB}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Atomic{Int}[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Vector{Pair{O,Int}}[], sz),
            Dict{Tuple{Int,O}, Int}(),
            sizehint!(A[], sz),
            sizehint!(Atomic{Int}[], sz),

            sizehint!(Array{B}(undef, 1), sz),
            sizehint!(Atomic{Int}[Atomic{Int}(0)], sz),
            sizehint!(Vector{Int}[Int[]], sz),
            sizehint!(Atomic{Int}[], sz),
            Dict{Tuple{Int,A}, Int}(),
            sizehint!(Array{O}(undef, 1), sz),

            ReentrantLock(),
            sizehint!(ReentrantLock[ReentrantLock()], sz),
            sizehint!(ReentrantLock[ReentrantLock()], sz),

            root_belief
        )
    end
end

##

struct TreeParallelPOWTreeObsNode{B,A,O,RB} <: BeliefNode
    tree::TreeParallelPOWTree{B,A,O,RB}
    node::Int
end

isroot(h::TreeParallelPOWTreeObsNode) = h.node == 1

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
