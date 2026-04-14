import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

class RobustPhaseRetrieval:
    """
    Implementation of Phase Retrieval using Subgradient Algorithm 
    via Scaled Polyak Step-size.
    """
    
    @staticmethod
    def generate_instance(n=200, m=5000, seed=123, normalize_rows=True, 
                          xstar_norm=1.0, outlier_ratio=0.0):
        """Generates the problem data A, b and ground truth x_star."""
        rng = np.random.default_rng(seed)

        A = rng.standard_normal((m, n))
        if normalize_rows:
            A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12

        x_star = rng.standard_normal(n)
        x_star = (x_star / (np.linalg.norm(x_star) + 1e-12)) * xstar_norm

        z = A @ x_star
        b = z * z

        if outlier_ratio > 0:
            num_outliers = int(outlier_ratio * m)
            outlier_indices = rng.choice(m, num_outliers, replace=False)
            b[outlier_indices] += rng.uniform(5, 15, size=num_outliers) 

        return A, b, x_star

    @staticmethod
    def err_pm(x: np.ndarray, x_star: np.ndarray) -> float:
        """Calculates error modulo sign: min(||x - x*||, ||x + x*||)."""
        return float(min(np.linalg.norm(x - x_star), np.linalg.norm(x + x_star)))

    def loss_and_subgradient(self, x: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
        """Computes L1 loss and the corresponding subgradient."""
        z = A @ x
        r = z * z - b
        m = A.shape[0]
        
        f = (1.0 / m) * float(np.linalg.norm(r, 1))
        g = (1.0 / m) * (A.T @ (2.0 * z * np.sign(r)))
        return f, g

    def solve_sg_sp(self, x0, A, b, x_star, f_star=0, sigma=1, max_iter=4000, tol=1e-12):
        """Subgradient Algorithm via Scaled Polyak Step-size."""
        x = x0.copy()
        f, g = self.loss_and_subgradient(x, A, b)

        hist = {"f": [float(f)], "err": [self.err_pm(x, x_star)]}

        for _ in range(int(max_iter)):
            gnorm = float(np.linalg.norm(g))
            if (not np.isfinite(gnorm)) or (gnorm <= tol):
                break
            
            step = (f - f_star) / (sigma * (gnorm**2))
            x_new = x - step * g

            f_new, g_new = self.loss_and_subgradient(x_new, A, b)
            if not np.isfinite(f_new):
                break

            x, f, g = x_new, float(f_new), g_new
            hist["f"].append(float(f))
            hist["err"].append(self.err_pm(x, x_star))

        return hist

def safe_ratio(seq, eps=1e-300):
    seq = np.asarray(seq, dtype=float)
    return seq[1:] / np.maximum(seq[:-1], eps)

def clip_ratios_to_err(hist, err_floor=1e-14):
    err = np.asarray(hist["err"], dtype=float)
    idx = np.where(err <= err_floor)[0]
    end = int(idx[0]) if idx.size > 0 else len(err) - 1
    return max(1, min(end, len(err) - 1))

def run_phase_transition():
    print("Starting Phase Transition Study...")
    solver = RobustPhaseRetrieval()
    n_val, n_trials = 1000, 20
    m_ratios = np.linspace(2, 5, 21)
    success_map = np.zeros(len(m_ratios))

    for i, ratio in enumerate(m_ratios):
        m_val = int(n_val * ratio)
        successes = 0
        for _ in range(n_trials):
            A, b, x_star = solver.generate_instance(n=n_val, m=m_val)
            x0 = np.random.standard_normal(n_val)
            hist = solver.solve_sg_sp(x0, A, b, x_star, max_iter=500)
            if (hist["err"][-1] / np.linalg.norm(x_star)) < 1e-5:
                successes += 1
        success_map[i] = successes / n_trials
        print(f"Ratio {ratio:.2f}: Success Rate {success_map[i]*100:.1f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(m_ratios, success_map, 'o-', linewidth=2, color='#2c3e50')
    plt.fill_between(m_ratios, success_map, alpha=0.2, color='#3498db')
    plt.xlabel(r"$m/n$")
    plt.ylabel("Probability of Success")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()

if __name__ == "__main__":
    # --- Setup ---
    n, alphas = 5000, [3.0, 3.5, 4.0, 4.5, 5.0]
    max_iter, sigma, tol = 1000, 1, 1e-12
    solver = RobustPhaseRetrieval()
    
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    results = {}

    print(f"Running SG_SP for n={n} and alphas={alphas}...")

    # --- Execution ---
    for alpha in alphas:
        m = int(alpha * n)
        A, b, x_star = solver.generate_instance(n=n, m=m, seed=123)
        hist = solver.solve_sg_sp(x0, A, b, x_star, sigma=sigma, max_iter=max_iter, tol=tol)
        
        L = clip_ratios_to_err(hist)
        results[alpha] = {
            "f": hist["f"], "err": hist["err"],
            "rf": safe_ratio(hist["f"])[:L], "re": safe_ratio(hist["err"])[:L], "L": L
        }
        print(f"Done for alpha = {alpha}")

    # --- Plotting (Exactly as your original design) ---
    styles = ['-', '--', '-.', ':', (0, (5, 1))]
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Figure 1: f(x_k)
    plt.figure(figsize=(8, 6))
    for i, alpha in enumerate(alphas):
        plt.semilogy(results[alpha]["f"], linestyle=styles[i], color=colors[i], 
                     linewidth=2, label=rf"$\alpha = {alpha}$")
    plt.xlabel("Iteration"); plt.ylabel(r"$f(x_k)$")
    plt.grid(True, which="both", linestyle='--', alpha=0.5); plt.legend()

    # Figure 2: error e_k
    plt.figure(figsize=(8, 6))
    for i, alpha in enumerate(alphas):
        plt.semilogy(results[alpha]["err"], linestyle=styles[i], color=colors[i], 
                     linewidth=2, label=rf"$\alpha = {alpha}$")
    plt.xlabel("Iteration"); plt.ylabel(r"$e_k=\min(\|x_k-x^\star\|,\|x_k+x^\star\|)$")
    plt.grid(True, which="both", linestyle='--', alpha=0.5); plt.legend()

    # Figure 3 & 4 (Ratios)
    alphas_2 = [4.0, 4.5, 5.0]
    for fig_type, key in [ (3, "rf"), (4, "re") ]:
        plt.figure(figsize=(8, 6))
        for i, alpha in enumerate(alphas_2):
            L = results[alpha]["L"]
            marker_every = max(1, L // 15)
            plt.plot(np.arange(L), results[alpha][key], linestyle=styles[i], 
                     marker=markers[i], color=colors[i], markevery=marker_every, 
                     linewidth=1.6, label=rf"$\alpha = {alpha}$")
        plt.xlabel("Iteration")
        plt.ylabel(r"$f_{k+1}/f_k$" if key=="rf" else r"$e_{k+1}/e_k$")
        plt.grid(True, linestyle='--', alpha=0.5); plt.legend()

    plt.tight_layout()
    run_phase_transition()
    plt.show()
