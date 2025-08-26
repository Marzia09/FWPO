# FWPO
# Apply to test ANN problem
dim = 3
lb = np.array([8, 4, 1e-4])
ub = np.array([128, 64, 1e-2])
best_params, best_mse, curve = gwo_fwpo_adaptive(fitness_ann, dim, lb, ub)

# Plot convergence
plt.plot(curve, label='GWO-FWPO Adaptive')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Convergence Curve with Adaptive τ and J')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_params, best_mse
