function simulate(pomcp::TreeParallelPOWPlanner, h_node::TreeParallelPOWTreeObsNode, s, d::Int, rng::AbstractRNG)
    # println("begin simulate")
    h = h_node.node
    sol = pomcp.solver
    tree = h_node.tree
    problem = pomcp.problem
    γ = POMDPs.discount(pomcp.problem)

    # println("begin Terminal Check")
    if POMDPs.isterminal(problem, s) || d ≤ 0
        return 0.0
    end

    if sol.enable_action_pw
        # println("Begin Action PW")
        total_n = tree.total_n[h][]
        # @show total_n
        if length(tree.tried[h]) ≤ sol.k_action*total_n^sol.alpha_action
            # println("Action PW criteria met")
            a = rand(rng, actions(problem))
            # @show a
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a, 0, 0.0, sol.check_repeat_act)
            end
            # println("Ln 25")
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            anode = length(tree.n)
            for a in actions(problem)
                push_anode!(tree, h, a, 0, 0.0, false)
            end
        end
    end
    # println("Finish Action Widening")

    best_node = select_best(tree, pomcp.criterion, h_node, rng)
    a = tree.a_labels[best_node]

    new_node = false
    if tree.n_a_children[best_node][] ≤ sol.k_observation*tree.n[best_node][]^sol.alpha_observation

        sp, o, r = @gen(:sp, :o, :r)(problem, s, a, rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = push_bnode!(
                pomcp.node_sr_belief_updater,
                problem,
                tree,
                best_node,
                s, a, sp, o, r,
                sol.check_repeat_obs
            )
        end
        lock(tree.tree_lock)
            lock.(tree.a_locks)
                push!(tree.generated[best_node], o=>hao)
            unlock.(tree.a_locks)
        unlock(tree.tree_lock)
    else

        sp, r = @gen(:sp, :r)(problem, s, a, rng)

    end
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
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node][]
    end
    unlock(tree.a_locks[best_node])

    # println("End Simulation")
    return R
end