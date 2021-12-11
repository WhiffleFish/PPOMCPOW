function select_best(tree::TreeParallelPOWTree, crit::MaxUCB, h_node::TreeParallelPOWTreeObsNode, rng::AbstractRNG)
    h = h_node.node
    iszero(tree.total_n[h]) && return rand(rng, tree.tried[h])
    best_ucb = -Inf
    c = crit.c
    best_node = 0
    tied = Int[]

    logtn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node][]
        q = tree.v[node]
        ucb = q + c*sqrt(logtn/n)
        if ucb > best_ucb
            best_ucb = ucb
            empty!(tied)
            push!(tied, node)
        elseif ucb == best_ucb
            push!(tied, node)
        end
    end

    if length(tied) === 1
        return only(tied)
    else
        return rand(rng, tied)
    end
end


function select_best(tree::TreeParallelPOWTree, crit::MaxQ, h_node::TreeParallelPOWTreeObsNode, rng::AbstractRNG)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_v = tree.v[best_node]
    @assert !isnan(best_v)
    for node in tree.tried[h][2:end]
        if tree.v[node] >= best_v
            best_v = tree.v[node]
            best_node = node
        end
    end
    return best_node
end


function select_best(tree::TreeParallelPOWTree, crit::MaxTries, h_node::TreeParallelPOWTreeObsNode, rng::AbstractRNG)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_n = tree.n[best_node][]
    @assert !isnan(best_n)
    for node in tree.tried[h][2:end]
        if tree.n[node][] >= best_n
            best_n = tree.n[node][]
            best_node = node
        end
    end
    return best_node
end


struct VirtualLoss
    c::Float64
end

function select_best(tree::TreeParallelPOWTree, crit::VirtualLoss, h_node::TreeParallelPOWTreeObsNode, rng::AbstractRNG)
    h = h_node.node
    iszero(tree.total_n[h]) && return rand(rng, tree.tried[h])
    best_ucb = -Inf
    c = crit.c
    best_node = 0
    tied = Int[]

    logtn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node][] + tree.o[node][]
        q = tree.v[node]
        ucb = q + c*sqrt(logtn/n)
        if ucb > best_ucb
            best_ucb = ucb
            empty!(tied)
            push!(tied, node)
        elseif ucb == best_ucb
            push!(tied, node)
        end
    end

    if length(tied) === 1
        return only(tied)
    else
        return rand(rng, tied)
    end
end
