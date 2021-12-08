function push_all_actions!(
        problem::POMDP,
        tree::TreeParallelPOWTree{B,A,O},
        h::Int) where {B,A,O}

    lock(tree.tree_lock)
    if isempty(tree.tried[h])
        ACT = actions(problem)
        L = length(ACT)
        init_idx = length(tree.n)
        lock.(tree.a_locks)
        lock.(tree.b_locks)
            for (i,a) in enumerate(ACT)
                anode = init_idx + i
                push!(tree.a_locks, ReentrantLock())
                push!(tree.n, Atomic{Int}(0))
                push!(tree.v, 0.0)
                push!(tree.generated, Pair{O,Int}[])
                push!(tree.a_labels, a)
                push!(tree.n_a_children, Atomic{Int}(0))

                push!(tree.tried[h], anode)
            end
        unlock.(tree.b_locks)
        unlock.(tree.a_locks[1:end-L])
    end
    unlock(tree.tree_lock)
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
    N = tree.total_n[h][]
    if length(tree.tried[h]) ≤ k_a*N^α_a
        anode = length(tree.n) + 1
        a = rand(rng, actions(problem))
        if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
            lock(tree.tree_lock)
                lock.(tree.a_locks)
                    push!(tree.a_locks, ReentrantLock())
                    push!(tree.n, Atomic{Int}(0))
                    push!(tree.v, 0.0)
                    push!(tree.generated, Pair{O,Int}[])
                    push!(tree.a_labels, a)
                    push!(tree.n_a_children, Atomic{Int}(0))
                    update_lookup && (tree.o_child_lookup[(h, a)] = anode)
                unlock.(tree.a_locks)

                for (i,l) in enumerate(tree.b_locks)
                    i ≠ h && lock(l) # i == h already locked
                end
                    push!(tree.tried[h], anode)
                unlock.(tree.b_locks)
            unlock(tree.tree_lock)
        end
    else
        unlock(tree.b_locks[h])
    end
end


function push_belief_pw!(
        pomcp,
        tree,
        best_node, s, a, rng
        )

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
            hao = tree.a_child_lookup[(best_node, o)]

            lock(tree.tree_lock)
                lock.(tree.a_locks)
                    push!(tree.generated[best_node], o=>hao)
                unlock.(tree.a_locks)
            unlock(tree.tree_lock)

        else
            new_node = true
            lock(tree.tree_lock)
            lock.(tree.b_locks)
                hao = length(tree.sr_beliefs) + 1
                push!(tree.sr_beliefs,
                      init_node_sr_belief(sol.node_sr_belief_updater,
                                          problem, s, a, sp, o, r))
                push!(tree.total_n, Atomic{Int}(0))
                push!(tree.tried, Int[])
                push!(tree.o_labels, o)
                check_repeat_obs && (tree.a_child_lookup[(best_node, o)] = hao)
                push!(tree.b_locks, ReentrantLock())
            unlock.(tree.b_locks[1:end-1])

            for (i,l) in enumerate(tree.a_locks)
                i ≠ best_node && lock(l) # i == best_node already locked
            end
                push!(tree.generated[best_node], o=>hao)
            unlock.(tree.a_locks)

            unlock(tree.tree_lock)
            atomic_add!(tree.n_a_children[best_node], 1)
        end
    else
        unlock(tree.a_locks[best_node])
        sp, r = @gen(:sp, :r)(problem, s, a, rng)
    end

    return sp, r, new_node
end
