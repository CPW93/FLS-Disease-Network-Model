# -*- coding: utf-8 -*-
"""
@author: Chinthaka
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, dok_matrix
from matplotlib.patches import Polygon, Patch

# ========================== Timer ==========================
start_time = time.time()

# ===================== Field / Network =====================
def create_coordinates(x, y):
    return [(i, j) for i in range(x) for j in range(y)]

def create_graphs_with_gaps(dimensions, gap, x_scale, y_scale):
    
    all_coordinates = []
    num_graphs_per_row = 3
    current_y = 0
    for row in range(2):
        current_x = 0
        for col in range(num_graphs_per_row):
            index = row * num_graphs_per_row + col
            x, y = dimensions[index]
            coords = create_coordinates(x, y)
            coords = [(c[0] * x_scale + current_x, c[1] * y_scale + current_y) for c in coords]
            all_coordinates.append(coords)
            current_x = coords[-1][0] + gap[0]
        current_y = all_coordinates[-1][-1][1] + gap[1]
    return all_coordinates

# Function to create a combined distance matrix
def create_combined_distance_matrix(coordinates):
    all_points = np.vstack(coordinates)
    tree = KDTree(all_points)
    num_points = len(all_points)
    combined_D = dok_matrix((num_points, num_points), dtype=np.float32)

    for i, point in enumerate(all_points):
        distances, indices = tree.query(point, k=num_points)
        combined_D[i, indices] = distances

    return combined_D.toarray()

def create_adjacency_matrix(distance_matrix):
    num_vertices = len(distance_matrix)
    row_idx, col_idx = np.where(distance_matrix > 0)
    adjacency_matrix = csr_matrix((np.ones(len(row_idx), dtype=int), (row_idx, col_idx)), shape=(num_vertices, num_vertices))
    return adjacency_matrix


def get_susceptible_to_infected_distances(distance_matrix, susceptible_indices, infected_indices):
    return distance_matrix[np.ix_(susceptible_indices, infected_indices)]


def build_node_to_plot_map(coordinates):
    """
    Map each node index (stacked vstack order) -> plot index (0..5).
    """
    node_to_plot = []
    for p_idx, coords in enumerate(coordinates):
        node_to_plot.extend([p_idx] * len(coords))
    return np.array(node_to_plot, dtype=int)

def build_plot_practice():
    """
    Top row plots 0,1,2 = 'till'; bottom row 3,4,5 = 'no-till'.
    """
    return np.array(['till','till','till','no-till','no-till','no-till'], dtype=object)

# ===================== SEIRB Simulator =====================
def SEIRB_network_tillage(
    G, A, timesteps,
    theta, beta_non, rho_beta, sigma, gamma, xi, r, k, tau_non, rho_tau,
    initial_infectednodes, combined_distance_matrix, d_threshold,
    node_to_plot, plot_practice, B0_non=4000.0, B0_till=4000.0
):
    N = G.number_of_nodes()
    P = len(np.unique(node_to_plot))

    # Per-plot params driven by practice
    beta_p = np.zeros(P, dtype=float)
    tau_p  = np.zeros(P, dtype=float)
    Bp     = np.zeros(P, dtype=float)
    for p in range(P):
        if plot_practice[p] == 'till':
            beta_p[p] = beta_non * rho_beta
            tau_p[p]  = tau_non  * rho_tau
            Bp[p]     = B0_till
        else:
            beta_p[p] = beta_non
            tau_p[p]  = tau_non
            Bp[p]     = B0_non

    node_is_till = (plot_practice[node_to_plot] == 'till')

    # States: 0=S, 1=E, 2=I, 3=R
    node_states = np.zeros(N, dtype=int)
    node_states[initial_infectednodes] = 2

    status_matrix = np.zeros((timesteps, 4), dtype=float)
    B_history     = np.zeros((timesteps, P), dtype=float)

    # Per-practice infected counts
    I_till   = np.zeros(timesteps, dtype=int)
    I_notill = np.zeros(timesteps, dtype=int)

    # t=0 record
    status_matrix[0] = [
        np.sum(node_states == 0),
        np.sum(node_states == 1),
        np.sum(node_states == 2),
        np.sum(node_states == 3)
    ]
    B_history[0] = Bp
    inf_mask0 = (node_states == 2)
    I_till[0]   = int(np.sum(inf_mask0 & node_is_till))
    I_notill[0] = int(np.sum(inf_mask0 & (~node_is_till)))

    for t in range(1, timesteps):
        infected_mask    = (node_states == 2)
        susceptible_idx  = np.where(node_states == 0)[0]
        infected_idx     = np.where(infected_mask)[0]

        Dsi = get_susceptible_to_infected_distances(combined_distance_matrix, susceptible_idx, infected_idx)
        if Dsi.size:
            has_close_infected = np.any(Dsi <= d_threshold, axis=1)
        else:
            has_close_infected = np.zeros(len(susceptible_idx), dtype=bool)

        infected_neighbors = A @ infected_mask
        local_risk = theta * infected_neighbors

        soil_term = beta_p[node_to_plot] * Bp[node_to_plot]

        lambda_total = local_risk + soil_term
        p_inf = 1.0 - np.exp(-lambda_total[susceptible_idx])

        if len(susceptible_idx):
            rand_vals = np.random.rand(len(susceptible_idx))
            newly_exposed_idx = susceptible_idx[(rand_vals < p_inf) & has_close_infected]
        else:
            newly_exposed_idx = np.array([], dtype=int)

        p_inc = 1.0 - np.exp(-sigma)
        p_rec = 1.0 - np.exp(-gamma)

        exposed_mask   = (node_states == 1)
        to_infected    = exposed_mask & (np.random.rand(N) < p_inc)
        to_recovered   = infected_mask & (np.random.rand(N) < p_rec)

        node_states[newly_exposed_idx] = 1
        node_states[to_infected]       = 2
        node_states[to_recovered]      = 3

        # Soil update (logistic growth - decay + shedding)
        I_counts    = np.bincount(node_to_plot[(node_states == 2)], minlength=P)
        growth      = r * Bp * (1.0 - Bp / k)
        decay       = tau_p * Bp
        contribution= xi * I_counts
        Bp = np.maximum(Bp + growth - decay + contribution, 0.0)

        status_matrix[t] = [
            np.sum(node_states == 0),
            np.sum(node_states == 1),
            np.sum(node_states == 2),
            np.sum(node_states == 3)
        ]
        B_history[t] = Bp

        inf_mask = (node_states == 2)
        I_till[t]   = int(np.sum(inf_mask & node_is_till))
        I_notill[t] = int(np.sum(inf_mask & (~node_is_till)))

    return status_matrix, B_history, I_till, I_notill

def SEIR_model(
    timesteps, theta, beta_non, rho_beta, sigma, gamma, xi, r, k, tau_non, rho_tau,
    initial_infectednodes, d_threshold, node_to_plot, plot_practice
):
    status_matrix, B_history, *_ = SEIRB_network_tillage(
        G, A, timesteps, theta, beta_non, rho_beta, sigma, gamma, xi,
        r, k, tau_non, rho_tau, initial_infectednodes, combined_distance_matrix,
        d_threshold, node_to_plot, plot_practice, B0_non=4000.0, B0_till=4000.0
    )
    return status_matrix


# =================== Posterior & Runner ====================
# Load posterior values
def read_posterior_values(file_path):
    posterior_values = np.loadtxt(file_path, delimiter=' ')
    if posterior_values.shape[1] != 7:
        raise ValueError(
            "Posterior file must have exactly 7 columns: theta, beta_non, xi, tau_non, rho_beta, rho_tau, d_threshold")
    return posterior_values


def run_simulation(posterior_values, timesteps, initial_infectednodes ):
    
    """
    For each posterior row, run 10 stochastic sims.
    Returns:
      sim_results: (R, T)
      sim_till:  (R, T)
      sim_not:   (R, T)
      param_list:(R, 7)
    where R = 10 * (# rows of posterior_values).
    """
    sim_results = []
    sim_till  = []
    sim_not = []
    param_list = []

    for theta, beta_non, xi, tau_non, rho_beta, rho_tau, d_threshold in posterior_values:
        for _ in range(10):  # 10 runs per parameter set
            status_matrix, *_ , I_till, I_notill= SEIRB_network_tillage( G, A, timesteps, theta, beta_non,            
                rho_beta, sigma, gamma, xi,                  
                r, k, tau_non, rho_tau, initial_infectednodes, combined_distance_matrix, 
                d_threshold, node_to_plot, plot_practice, B0_non=4000.0, B0_till=4000.0       
                )
            sim_results.append(status_matrix[:, 2])  # Infected only (total)
            sim_till.append(I_till)                # infected in tilled (tilled plot)
            sim_not.append(I_notill)                # infected in non-tilled (non-tilled plot)
            param_list.append([theta, beta_non,  xi, tau_non,  rho_beta, rho_tau, d_threshold])
    
    return np.array(sim_results), np.array(sim_till), np.array(sim_not), np.array(param_list)


# ================== Build field once =======================
# Parameters
dimensions = [(72, 4)] * 6
gaps, x_scale, y_scale = [150, 150], 8.33, 76.2

coordinates = create_graphs_with_gaps(dimensions, gaps, x_scale, y_scale)
combined_distance_matrix = create_combined_distance_matrix(coordinates)
combined_adjacency_matrix = create_adjacency_matrix(combined_distance_matrix)
G = nx.from_numpy_array(combined_adjacency_matrix)
A = nx.to_scipy_sparse_array(G)

node_to_plot = build_node_to_plot_map(coordinates)
plot_practice = build_plot_practice()

sigma, gamma, r, k = 1/10, 1/75, 0.001, 60000
timesteps = 139
                                  
# Load posterior values and run simulations

# ================== Data & file paths ======================
# Update these two paths for your machine
initial_infectednodes = np.loadtxt(
    r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\initial_infected_nodes_cluster_till_notill_random_dist_euclidean2026Feb1.txt",
    dtype=int
).tolist()

posterior_file = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\till_notill_random2\posterior_samples_cluster_till_notill_random_cluster_till_notill.txt"

# Observed total infected at 8 days
observed_timesteps = [0, 45, 50, 75, 89, 96, 117, 138]
observed_data      = np.array([52, 86, 104, 138, 276, 362, 484, 622])

# ================== Run sims & select best =================
posterior_values = read_posterior_values(posterior_file)

sim_results, sim_till, sim_not, param_list = run_simulation(
    posterior_values, timesteps, initial_infectednodes
)

# Score each run against observed totals at 8 timepoints
distance_dict = {}
for i, sim_curve in enumerate(sim_results):
    extracted = sim_curve[observed_timesteps]
    distance  = np.linalg.norm(extracted - observed_data)
    distance_dict[i] = distance

# Best 100 runs (indices into the big run pool)
n_best = 100
best_indices = sorted(distance_dict, key=distance_dict.get)[:n_best]

best_total  = sim_results[best_indices]   # (100, T)
best_till   = sim_till [best_indices]   # (100, T)
best_not    = sim_not  [best_indices]   # (100, T)
best_params = param_list[best_indices]  # (100, 7)

print("Best pool shapes:", best_total.shape, best_till.shape, best_not.shape)

# ===================== Summaries & fits ====================
tvec = np.arange(timesteps)

# Totals
mean_total = best_total.mean(axis=0)
sd_total   = best_total.std(axis=0, ddof=1)
mean_obs_points = mean_total[observed_timesteps]

print("\nMean at observed times:", mean_obs_points)
dist_abs = np.sum(np.sqrt((mean_obs_points - observed_data)**2))
dist_euc = np.sqrt(np.sum((mean_obs_points - observed_data)**2))
print("Abs sum sqrt diffs:", dist_abs)
print("Euclidean distance:", dist_euc)


mean_till = best_till.mean(axis=0);  sd_till = best_till.std(axis=0, ddof=1)
mean_not  = best_not.mean(axis=0);   sd_not  = best_not.std(axis=0, ddof=1)

# ========================= Plots ===========================

# Tilled vs No-till (from best runs)
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
})

ax.set_title("Posterior-Predictive Infection Trajectories", fontsize=18)  # one-off override

# --- Colors (colorblind-safe) ---
tilled_line = "#0072B2"   # blue
tilled_band = "#A6CEE3"   # light desaturated blue for band
notill_line = "#E69F00"   # orange
notill_band = "#FDD0A2"   # light desaturated orange for band

# --- TILLED ---
tilled_lower = mean_till - 1.96 * sd_till
tilled_upper = mean_till + 1.96 * sd_till

lt, = ax.plot(
    tvec, mean_till, label="Till (mean)",
    lw=2.2, color=tilled_line, zorder=3
)
ax.fill_between(
    tvec, tilled_lower, tilled_upper,
    facecolor=tilled_band, alpha=0.45, linewidth=0, zorder=1
)

# --- NO-TILL ---
not_lower = mean_not - 1.96 * sd_not
not_upper = mean_not + 1.96 * sd_not

ln, = ax.plot(
    tvec, mean_not, label="No-till (mean)",
    lw=2.2, color=notill_line, marker='x', markersize=5,
    markevery=7, mew=1.2, zorder=3
)
ax.fill_between(
    tvec, not_lower, not_upper,
    facecolor=notill_band, alpha=0.30, linewidth=0, zorder=1
)

# --- Labels & Legend ---
ax.set_xlabel("Timesteps")
ax.set_ylabel("Infected plants")
ax.set_title("Posterior-Predictive Infection Trajectories: Till vs No-till")
ax.grid(True, alpha=0.35)

# Legend entries with band proxies

tilled_band_proxy = Patch(facecolor=tilled_band, alpha=0.60, label="Till 95% band")
notill_band_proxy = Patch(facecolor=notill_band, alpha=0.30, label="No-till 95% band")
ax.legend(
    handles=[lt, ln, tilled_band_proxy, notill_band_proxy],
    loc="upper left",       #  top-left corner
    frameon=False,          # remove legend box outline
)

plt.tight_layout()
plt.show()

# 1)  Total = Tilled + No-till in the selected runs?
np.allclose(best_total, best_till + best_not)  # should be True

# 2) see the negative correlation at a specific time (e.g., last time point)
t = -1
corr = np.corrcoef(best_till[:, t], best_not[:, t])[0, 1]
var_total = np.var(best_total[:, t], ddof=1)
var_parts = (np.var(best_till[:, t], ddof=1) +
             np.var(best_not[:, t],  ddof=1) +
             2*np.cov(best_till[:, t], best_not[:, t], ddof=1)[0,1])
print("corr(Till, No-till) =", corr)
print("Var(total) vs sum-of-parts =", var_total, var_parts)  # should match (up to rounding)

# ================= Parameter summaries =====================
mean_param_values = np.mean(best_params, axis=0)
print("\nFinal average parameters (mean of best simulations):")
print(f"Theta:        {mean_param_values[0]:.16f}")
print(f"Beta_non:     {mean_param_values[1]:.16f}")
print(f"xi:           {mean_param_values[2]:.16f}")
print(f"tau_non:      {mean_param_values[3]:.16f}")
print(f"rho_beta:     {mean_param_values[4]:.16f}")
print(f"rho_tau:      {mean_param_values[5]:.16f}")
print(f"d_threshold:  {mean_param_values[6]:.16f}")

best_param_set = best_params[0]
print("\nBest individual parameter set (lowest distance in the pool):")
print(f"Theta:        {best_param_set[0]:.16f}")
print(f"Beta_non:     {best_param_set[1]:.16f}")
print(f"xi:           {best_param_set[2]:.16f}")
print(f"tau_non:      {best_param_set[3]:.16f}")
print(f"rho_beta:     {best_param_set[4]:.16f}")
print(f"rho_tau:      {best_param_set[5]:.16f}")
print(f"d_threshold:  {best_param_set[6]:.16f}")

# ============== Optional: posterior histograms =============
parameter_names = [
    r'$\theta$',
    r'$\beta_{\mathrm{non}}$',
    r'$\xi$',
    r'$\tau_{\mathrm{non}}$',
    r'$\rho_{\beta}$',
    r'$\rho_{\tau}$',
    r'$d_{0}$',
]

plt.figure(figsize=(22,14))

colors = [
    "#2093C3"
   ]

for i in range(7):
    plt.subplot(3,3,i+1)
    plt.hist(posterior_values[:, i], bins=20,color=colors, edgecolor='black', alpha=0.75 )
    plt.xlabel(parameter_names[i], fontsize=28, fontweight='bold')
    plt.ylabel('Frequency', fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout(); plt.show()

print("\nTotal runtime (s):", time.time() - start_time)
