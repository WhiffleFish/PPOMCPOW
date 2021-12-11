function push_all_actions!(
        problem::POMDP,
        tree::TreeParallelPOWTree{B,A,O},
        h::Int) where {B,A,O}

    lock(tree.tree_lock)
    @debug("$(threadid()) - tree locked")
    if isempty(tree.tried[h])
        @debug("$(threadid()) - node $h empty")
        ACT = actions(problem)
        L = length(ACT)
        init_idx = length(tree.n)
        for (i,a) in enumerate(ACT)
                anode = init_idx + i
                @debug("$(threadid()) - pre a lock push")
                push!(tree.a_locks, SpinLock())
                @debug("$(threadid()) - pushed a lock")
                push!(tree.n, Atomic{Int}(0))
                @debug("$(threadid()) - pushed to tree.n")
                push!(tree.v, 0.0)
                @debug("$(threadid()) - pushed to tree.v")
                push!(tree.generated, Pair{O,Int}[])
                @debug("$(threadid()) - pushed to tree.generated")
                push!(tree.a_labels, a)
                @debug("$(threadid()) - pushed to tree.a_labels")
                push!(tree.n_a_children, Atomic{Int}(0))
                @debug("$(threadid()) - pushed to tree.n_a_children")

                push!(tree.tried[h], anode)
                @debug("$(threadid()) - pushed to tree.tried[h]")
                tree.n_tried[h] += 1
                @debug("$(threadid()) - added to tree.n_tried[h]")
            end
    end
    unlock(tree.tree_lock)
    @debug("$(threadid()) - tree unlocked")
end

function push_action_pw!(
    pomcp::TreeParallelPOWPlanner,
    tree::TreeParallelPOWTree{B,A,O},
    h::Int, rng::AbstractRNG) where {B,A,O}

    sol = pomcp.solver
    problem = pomcp.problem
    k_a, α_a = sol.k_action, sol.alpha_action
    update_lookup = sol.check_repeat_act

    lock(tree.b_locks[h])
    N = tree.total_n[h]
    if tree.n_tried[h] ≤ k_a*N^α_a
        anode = length(tree.n) + 1
        a = rand(rng, actions(problem))
        if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
            tree.n_tried[h] +=  1
            unlock(tree.b_locks[h])

            lock(tree.tree_lock)
                    push!(tree.a_locks, SpinLock())
                    push!(tree.n, Atomic{Int}(0))
                    push!(tree.v, 0.0)
                    push!(tree.generated, Pair{O,Int}[])
                    push!(tree.a_labels, a)
                    push!(tree.n_a_children, Atomic{Int}(0))
                    update_lookup && (tree.o_child_lookup[(h, a)] = anode)

                    push!(tree.tried[h], anode)
            unlock(tree.tree_lock)
        else
            unlock(tree.b_locks[h])
        end
    else
        unlock(tree.b_locks[h])
    end
end


function push_belief_pw!(
        pomcp::TreeParallelPOWPlanner,
        tree::TreeParallelPOWTree{B,A},
        best_node::Int, s, a::A, rng::AbstractRNG
        ) where {B,A}

    new_node = false
    sol = pomcp.solver
    problem = pomcp.problem
    check_repeat_obs = sol.check_repeat_obs
    k_o, α_o = sol.k_observation, sol.alpha_observation

    lock(tree.a_locks[best_node])
    N = tree.n[best_node][]
    Na = tree.n_a_children[best_node][]
    if Na ≤ k_o*N^α_o
        sp, o, r = @gen(:sp, :o, :r)(problem, s, a, rng)

        if check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            unlock(tree.a_locks[best_node])
            hao = tree.a_child_lookup[(best_node, o)]

            lock(tree.tree_lock)
                    push!(tree.generated[best_node], o=>hao)
            unlock(tree.tree_lock)

        else
            new_node = true
            unlock(tree.a_locks[best_node])

            lock(tree.tree_lock)

                hao = length(tree.sr_beliefs) + 1
                push!(tree.sr_beliefs,
                      init_node_sr_belief(sol.node_sr_belief_updater,
                                          problem, s, a, sp, o, r))
                push!(tree.total_n, 0)
                push!(tree.tried, Int[])
                push!(tree.n_tried, 0)
                push!(tree.o_labels, o)
                check_repeat_obs && (tree.a_child_lookup[(best_node, o)] = hao)
                push!(tree.b_locks, SpinLock())
                atomic_add!(tree.n_a_children[best_node], 1)
                push!(tree.generated[best_node], o=>hao)

            unlock(tree.tree_lock)

        end
    else
        isempty(tree.generated[best_node]) && @warn("generated empty after pw")
        unlock(tree.a_locks[best_node])
        sp, r = @gen(:sp, :r)(problem, s, a, rng)
    end

    return sp, r, new_node
end
