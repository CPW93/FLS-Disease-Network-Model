# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:53:26 2025

@author: Chinthaka
"""

import numpy as np

# === Load posterior samples ===
path = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\till_notill_random2\posterior_samples_cluster_till_notill_random_cluster_till_notill.txt"
posterior = np.loadtxt(path)

# Extract parameters (your columns 4 and 5)
rho_beta = np.asarray(posterior[:, 4], dtype=float)  # ρβ (exposure ratio)
rho_tau  = np.asarray(posterior[:, 5], dtype=float)  # ρτ (decay ratio)

# Keep finite samples only (robustness)
rho_beta = rho_beta[np.isfinite(rho_beta)]
rho_tau  = rho_tau[np.isfinite(rho_tau)]

def left_one_sided_CI_test(samples, alpha=0.05, null=1.0):
    """
    H0: theta = null vs H1: theta < null (left-tailed, level alpha)
    CI_{1-alpha} = (-inf, U), where U = Q_{1-alpha}(posterior)
    Reject H0 if U < null.
    """
    U = float(np.quantile(samples, 1.0 - alpha))
    reject = (U < null)
    # Posterior support for H1
    p = float(np.mean(samples < null))
    # Smallest alpha that would reject under this rule: require p > 1 - alpha -> alpha > 1 - p
    alpha_star = max(0.0, 1.0 - p)
    return {
        "U": U,
        "CI": (-np.inf, U),
        "reject": reject,
        "posterior_support": p,     # P(theta < null)
        "alpha_star": alpha_star    # minimal alpha to reject
    }

def right_one_sided_CI_test(samples, alpha=0.05, null=1.0):
    """
    H0: theta = null vs H1: theta > null (right-tailed, level alpha)
    CI_{1-alpha} = (L, inf), where L = Q_{alpha}(posterior)
    Reject H0 if L > null.
    """
    L = float(np.quantile(samples, alpha))
    reject = (L > null)
    # Posterior support for H1
    p = float(np.mean(samples > null))
    # Smallest alpha that would reject: require p > 1 - alpha -> alpha > 1 - p
    alpha_star = max(0.0, 1.0 - p)
    return {
        "L": L,
        "CI": (L, np.inf),
        "reject": reject,
        "posterior_support": p,     # P(theta > null)
        "alpha_star": alpha_star
    }

def print_report(name, test_fn, samples, alphas):
    print(f"\n=== {name} ===")
    # Always show posterior support and alpha* once
    tmp = test_fn(samples, alpha=0.05, null=1.0)
    if "U" in tmp:
        ci_str = f"(-∞, {tmp['U']:.4f}) at α=0.05"
    else:
        ci_str = f"({tmp['L']:.4f}, ∞) at α=0.05"
    print(f"Posterior support for H1: {tmp['posterior_support']:.4f} ; "
          f"minimal α to reject (α*): {tmp['alpha_star']:.4f} ; "
          f"CI@0.05: {ci_str}")

    for alpha in alphas:
        res = test_fn(samples, alpha=alpha, null=1.0)
        if "U" in res:
            print(f"α={alpha:>4.2f} → CI_(1-α)=(-∞, {res['U']:.4f}) ; "
                  f"Decision: {'REJECT H0' if res['reject'] else 'fail to reject'} "
                  f"(needs U < 1)")
        else:
            print(f"α={alpha:>4.2f} → CI_(1-α)=({res['L']:.4f}, ∞) ; "
                  f"Decision: {'REJECT H0' if res['reject'] else 'fail to reject'} "
                  f"(needs L > 1)")

# Hypothesis 1 (exposure): H0: θβ = 1 vs H1: θβ < 1 (left-tailed)
print_report(
    name="Testing rho_β < 1 (tillage reduces soil exposure)",
    test_fn=left_one_sided_CI_test,
    samples=rho_beta,
    alphas=[0.05]
)

# Hypothesis 2 (decay): H0: θτ = 1 vs H1: θτ > 1 (right-tailed)
print_report(
    name="Testing rho_τ > 1 (tillage speeds up residue decay)",
    test_fn=right_one_sided_CI_test,
    samples=rho_tau,
    alphas=[0.05]
)
