function simulate(pomcp::TreeParallelPOWPlanner, h_node::TreeParallelPOWTreeObsNode, s, d::Int, rng::AbstractRNG)
    h = h_node.node
    sol = pomcp.solver
    tree = h_node.tree
    problem = pomcp.problem
    γ = POMDPs.discount(pomcp.problem)

    if POMDPs.isterminal(problem, s) || d ≤ 0
        return 0.0
    end

    if sol.enable_action_pw
        push_action_pw!(pomcp, tree, h, rng)
    else
        push_all_actions!(problem, tree, h)
    end

    best_node = select_best(tree, pomcp.criterion, h_node, rng)
    a = tree.a_labels[best_node]

    sp, r, new_node = push_belief_pw!(pomcp, tree, best_node, s, a, rng)
    # println("Finish Observation Widening")

    if isinf(r)
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        R = r + γ*estimate_value(pomcp.solved_estimate, problem, sp, d-1, rng)
    else
        pair = rand(rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(rng, tree.sr_beliefs[hao])

        R = r + γ*simulate(pomcp, TreeParallelPOWTreeObsNode(tree, hao), sp, d-1, rng)
    end
    # println("Finish Observation Widening")

    atomic_add!(tree.n[best_node], 1)
    atomic_add!(tree.total_n[h], 1)
    lock(tree.a_locks[best_node])
    if tree.v[best_node] != -Inf # when would this ever be the case?
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node][]
    end
    unlock(tree.a_locks[best_node])

    # println("End Simulation")
    return R
end
