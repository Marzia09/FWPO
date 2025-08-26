# Enhanced GWO-FWPO with adaptive τ and J
def gwo_fwpo_adaptive(fit_fn, dim, lb, ub, pop_size=30, max_iter=50, tau_init=0.05, J_init=1.2):
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness_vals = np.array([fit_fn(ind) for ind in pop])
    convergence = []
    tau = tau_init
    J = J_init
    stagnation_counter = 0
    best_fitness_history = []

    for t in range(max_iter):
        sorted_idx = np.argsort(fitness_vals)
        alpha, beta, delta = pop[sorted_idx[0]], pop[sorted_idx[1]], pop[sorted_idx[2]]
        best_fitness = fitness_vals[sorted_idx[0]]
        convergence.append(best_fitness)
        best_fitness_history.append(best_fitness)

        # Adaptive τ using standard error
        tau = np.std(fitness_vals) / np.sqrt(pop_size)

        # Adaptive J: success-based adaptation
        if t > 1 and abs(convergence[-1] - convergence[-2]) < 1e-4:
            stagnation_counter += 1
            if stagnation_counter >= 2:
                J *= 1.05
        else:
            stagnation_counter = 0
            J *= 0.95
        J = np.clip(J, 0.5, 2.0)

        # Update positions
        a = 2 * (1 - t / (max_iter - 1))
        new_pop = np.copy(pop)
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha - pop[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta - pop[i])
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta - pop[i])
            X3 = delta - A3 * D_delta

            X_new = (X1 + X2 + X3) / 3
            f_new = fit_fn(X_new)

            if f_new <= fitness_vals[i] + tau:
                new_pop[i] = X_new
                fitness_vals[i] = f_new
            else:
                epsilon = np.random.uniform(-0.01, 0.01, dim)
                new_pop[i] = pop[i] + J * (pop[i] - X_new) + epsilon
                new_pop[i] = np.clip(new_pop[i], lb, ub)
                fitness_vals[i] = fit_fn(new_pop[i])
        pop = new_pop

    best_index = np.argmin(fitness_vals)
    return pop[best_index], fitness_vals[best_index], convergence

