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
        total_n = tree.total_n[h][]
        if length(tree.tried[h]) ≤ sol.k_action*total_n^sol.alpha_action
            a = rand(rng, actions(problem))
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, problem, TreeParallelPOWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, problem, TreeParallelPOWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, problem, TreeParallelPOWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, problem, TreeParallelPOWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h][]

    best_node = select_best(tree, pomcp.criterion, h_node, rng)
    a = tree.a_labels[best_node]

    new_node = false
    if tree.n_a_children[best_node][] ≤ sol.k_observation*tree.n[best_node][]^sol.alpha_observation

        sp, o, r = @gen(:sp, :o, :r)(problem, s, a, rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            push_bnode!(
                pomcp.node_sr_belief_updater,
                problem,
                tree,
                best_node,
                s, a, sp, o, r;
                sol.check_repeat_obs
            )
        end
        lock.(tree.a_locks)
            push!(tree.generated[best_node], o=>hao)
        unlock.(tree.a_locks)
    else

        sp, r = @gen(:sp, :r)(problem, s, a, rng)

    end

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

    Threads.atomic_add!(tree.n[best_node], 1)
    Threads.atomic_add!(tree.total_n[h], 1)
    lock(tree.a_locks[best_node])
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node][]
    end
    unlock(tree.a_locks[best_node])

    return R
end
