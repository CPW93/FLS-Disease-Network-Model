
import os
import networkx as nx
import random
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.stats._distn_infrastructure import rv_frozen, rv_sample
from scipy.stats import rv_discrete, rv_continuous
from scipy.spatial import KDTree
from scipy.sparse import dok_matrix

"""Create Network"""
###############################################################################
start_time = time.time()
# Function to create coordinates for a grid
def create_coordinates(x, y):
    return [(i, j) for i in range(x) for j in range(y)]

# Function to create multiple graphs with gaps
def create_graphs_with_gaps(dimensions, gap, x_scale, y_scale):
    all_coordinates = []
    num_graphs_per_row = 3
    current_y = 0  

    for row in range(2):  
        current_x = 0  
        for col in range(num_graphs_per_row):  
            index = row * num_graphs_per_row + col
            x, y = dimensions[index]
            coordinates = create_coordinates(x, y)
            coordinates = [(c[0] * x_scale + current_x, c[1] * y_scale + current_y) for c in coordinates]
            all_coordinates.append(coordinates)
            current_x = coordinates[-1][0] + gap[0]
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



# ---- Plot mapping (6 plots from create_graphs_with_gaps order) ----
def build_node_to_plot_map(coordinates):
    """
    coordinates: list of 6 lists; each list is the coords of one plot, in the order they were created.
    Returns:
        node_to_plot: array of length N giving the plot index (0,...,5) for each node in the vstack order
    """
    node_to_plot = []
    for p_idx, coords in enumerate(coordinates):
        node_to_plot.extend([p_idx] * len(coords))
    return np.array(node_to_plot, dtype=int)

# top row (row=0) -> plots 0,1,2 ; bottom row (row=1) -> plots 3,4,5
def build_plot_practice():
    """
    Return an array of length 6 with entries 'till' or 'no-till'.
    By design: top row = tilled (0,1,2), bottom row = no-till (3,4,5).
    """
    return np.array(['till','till','till','no-till','no-till','no-till'], dtype=object)

def SEIRB_network_tillage(
    G, A, timesteps, 
    theta,               # plant-to-plant (old theta)
    beta_non,            # soil-to-plant base for no-till
    rho_beta,          # multiplier for till plots: beta_till = beta_non * theta_beta  (expect < 1)
    sigma, gamma,        # E->I and I->R daily probs via 1-exp(-sigma), 1-exp(-gamma)
    xi,                  # contribution from infectious plants to soil 
    r, k,                # soil logistic growth
    tau_non,             # soil removal for no-till
    rho_tau,           # multiplier for till plots: tau_till = tau_non * theta_tau (expect > 1)
    initial_infectednodes,
    combined_distance_matrix, d_threshold,  # keep spatial gating
    node_to_plot,        # array length N -> plot index 0..5
    plot_practice,       # array length 6 -> 'till' or 'no-till'
    B0_non=4000.0,       # initial B for no-till plots
    B0_till=4000.0       # initial B for tilled plots (can set < no-till if you wish)
):
    N = G.number_of_nodes() #number of nodes in the graph
    P = len(np.unique(node_to_plot))  # should be 6

    # Per-plot params from practice
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

    # States: 0=S, 1=E, 2=I, 3=R
    node_states = np.zeros(N, dtype=int)
    node_states[initial_infectednodes] = 2

    status_matrix = np.zeros((timesteps, 4), dtype=float) # we can use int also but this is help to get avg and so
    B_history     = np.zeros((timesteps, P), dtype=float)

    # record t=0
    status_matrix[0] = [
        np.sum(node_states == 0),
        np.sum(node_states == 1),
        np.sum(node_states == 2),
        np.sum(node_states == 3)
    ]
    B_history[0] = Bp # contamination level

    for t in range(1, timesteps):
        infected_mask = (node_states == 2)
        susceptible_idx = np.where(node_states == 0)[0]

        # Spatial gate: only allow infection if a susceptible is within d_threshold of any infected
        infected_idx = np.where(infected_mask)[0]
        Dsi = get_susceptible_to_infected_distances(combined_distance_matrix, susceptible_idx, infected_idx)
        
        #Dsi = distance_s_to_i
        
        if Dsi.size:
            has_close_infected = np.any(Dsi <= d_threshold, axis=1)
        else:
            has_close_infected = np.zeros(len(susceptible_idx), dtype=bool)

        # Network (theta) term: infected neighbors count
        infected_neighbors = A @ infected_mask  # vector length N
        local_risk = theta * infected_neighbors

        # Soil (beta) term: use the soil B of the plot the node belongs to
        # index Bp by each node's plot
        soil_term = beta_p[node_to_plot] * Bp[node_to_plot]

        # Total hazard
        lambda_total = np.zeros(N, dtype=float)
        lambda_total += local_risk
        lambda_total += soil_term

        # Convert to per-day infection probability for susceptible nodes
        p_inf = 1.0 - np.exp(-lambda_total[susceptible_idx])
        # apply spatial gate
        if len(susceptible_idx):
            rand_vals = np.random.rand(len(susceptible_idx))
            newly_exposed_idx = susceptible_idx[(rand_vals < p_inf) & has_close_infected]
        else:
            newly_exposed_idx = np.array([], dtype=int)

        # E->I and I->R (daily hazard to prob)
        p_inc = 1.0 - np.exp(-sigma)
        p_rec = 1.0 - np.exp(-gamma)

        # apply transitions
        exposed_mask   = (node_states == 1)
        to_infected    = exposed_mask & (np.random.rand(N) < p_inc)
        to_recovered   = infected_mask & (np.random.rand(N) < p_rec)

        node_states[newly_exposed_idx] = 1
        node_states[to_infected]       = 2
        node_states[to_recovered]      = 3

        # ----- Soil update per plot -----
        # Infectious counts per plot
        I_counts = np.bincount(node_to_plot[infected_mask], minlength=P)

        # Discrete-time update
        growth      = r * Bp * (1.0 - Bp / k)
        decay       = tau_p * Bp
        contribution= xi * I_counts
        Bp = np.maximum(Bp + growth - decay + contribution, 0.0)

        # record
        status_matrix[t] = [
            np.sum(node_states == 0),
            np.sum(node_states == 1),
            np.sum(node_states == 2),
            np.sum(node_states == 3)
        ]
        B_history[t] = Bp

    return status_matrix, B_history


def SEIR_model(timesteps, theta, beta_non,            
    rho_beta, sigma, gamma, xi,                  
    r, k, tau_non, rho_tau, initial_infectednodes, d_threshold, node_to_plot, plot_practice):

    status_matrix, B_history = SEIRB_network_tillage( G, A, timesteps, theta, beta_non,            
        rho_beta, sigma, gamma, xi,                  
        r, k, tau_non, rho_tau, initial_infectednodes, combined_distance_matrix, 
        d_threshold, node_to_plot, plot_practice, B0_non=4000.0, B0_till=4000.0       
        )

    return status_matrix

# Load posterior values
def read_posterior_values(file_path):
    posterior_values = np.loadtxt(file_path, delimiter=' ')
    if posterior_values.shape[1] != 7:
        raise ValueError(
            "Posterior file must have exactly 7 columns: theta, beta_non, xi, tau_non, rho_beta, rho_tau, d_threshold")
    return posterior_values

# Run SEIR simulation and track parameters
def run_simulation(posterior_values, timesteps, initial_infectednodes 
                  ):
    sim_results = []
    param_list = []

    for theta, beta_non, xi, tau_non, rho_beta, rho_tau, d_threshold in posterior_values:
        for _ in range(10):  # 10 runs per parameter set
            status_matrix, *_ = SEIRB_network_tillage( G, A, timesteps, theta, beta_non,            
                rho_beta, sigma, gamma, xi,                  
                r, k, tau_non, rho_tau, initial_infectednodes, combined_distance_matrix, 
                d_threshold, node_to_plot, plot_practice, B0_non=4000.0, B0_till=4000.0       
                )
            sim_results.append(status_matrix[:, 2])  # Infected only
            param_list.append([theta, beta_non,  xi, tau_non,  rho_beta, rho_tau, d_threshold])
    
    return np.array(sim_results), np.array(param_list)


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
initial_infectednodes = np.loadtxt(r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\till_notill_random2\initial_infected_nodes_cluster_till_notill_random.txt",
                                   dtype=int).tolist()
 
# Observed data
observed_timesteps = [0, 45, 50, 75, 89, 96, 117, 138]
observed_data = np.array([52, 86, 104, 138, 276, 362, 484, 622])

# Load posterior values and run simulations
posterior_file = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\posterior_samples_cluster_till_notill_random_dist_euclidean2026Feb1_cluster_till_notill.txt"
posterior_values = read_posterior_values(posterior_file)

# Run simulations and collect parameter sets
sim_results, param_list = run_simulation(
    posterior_values, timesteps, initial_infectednodes
)

# Compute distances
distance_dict = {}
for i, sim_data in enumerate(sim_results):
    extracted_data = sim_data[observed_timesteps]
    distance = np.linalg.norm(extracted_data - observed_data)
    distance_dict[i] = distance

# Select best 100
n = 100
best_indices = sorted(distance_dict, key=distance_dict.get)[:n]
best_simulations = sim_results[best_indices]

print(best_simulations.shape)

best_params = param_list[best_indices]

# Compute mean of simulations
mean_sim_data = np.mean(best_simulations, axis=0)
std_sim_data = np.std(best_simulations, axis=0)

mean_sim_timesteps_8 = mean_sim_data[observed_timesteps]

print(mean_sim_timesteps_8)

dist_abs = np.sum(np.sqrt((mean_sim_timesteps_8 - observed_data)**2))

print(dist_abs)

dist_euc = np.sqrt(np.sum((mean_sim_timesteps_8 - observed_data)**2))

print(dist_euc)

# Plot
plt.figure(figsize=(10, 6))
timesteps_arr = np.arange(timesteps)

plt.plot(timesteps_arr, mean_sim_data, label="Mean Simulated Data", color='blue')
plt.scatter(observed_timesteps, observed_data, label="Observed Data", color='red', zorder=3)
plt.fill_between(timesteps_arr, mean_sim_data - 1.96*std_sim_data,
                 mean_sim_data + 1.96*std_sim_data, 
                 color='blue', alpha=0.2, label="Confidence Band (±1.96 Std)")

plt.xlabel("Timesteps")
plt.ylabel("Number of Infected Nodes")
plt.title("Observed Data vs Simulated Data")
plt.legend()
plt.grid(True)
plt.show()

# Compute and print final parameters
mean_param_values = np.mean(best_params, axis=0)
print("\nFinal average parameters (mean of best simulations):")
print(f"Theta:        {mean_param_values[0]:.16f}")
print(f"Beta_non:         {mean_param_values[1]:.16f}")
print(f"xi:         {mean_param_values[2]:.16f}")
print(f"tau_non:         {mean_param_values[3]:.16f}")
print(f"rho_beta:  {mean_param_values[4]:.16f}")
print(f"rho_tau:         {mean_param_values[5]:.16f}")
print(f"d_threshold:  {mean_param_values[6]:.16f}")

# Print the single best parameter set (lowest distance)
best_index = best_indices[0]           # make sure this is an int index into param_list
best_param_set = param_list[best_index]
print("\nBest individual parameter set (lowest distance):")
print(f"Theta:        {best_param_set[0]:.16f}")
print(f"Beta_non:     {best_param_set[1]:.16f}")
print(f"xi:           {best_param_set[2]:.16f}")
print(f"tau_non:      {best_param_set[3]:.16f}")
print(f"rho_beta:     {best_param_set[4]:.16f}")
print(f"rho_tau:      {best_param_set[5]:.16f}")
print(f"d_threshold:  {best_param_set[6]:.16f}")

# Plot histograms of posterior values
parameter_names = [r'$\theta$',                # Theta
r'$\beta_{\mathrm{non}}$',  # Beta_non
r'$\xi$',                   # xi
r'$\tau_{\mathrm{non}}$',   # tau_non
r'$\rho_{\beta}$',          # rho_beta
r'$\rho_{\tau}$',           # rho_tau
r'$d_{\mathrm{th}}$',       # d_threshold]
]

plt.figure(figsize=(14, 10))  # increase figure size
for i in range(7):
    plt.subplot(3, 3, i + 1)  # arrange in 3x3 grid (enough for 7 plots)
    plt.hist(posterior_values[:, i], bins=20, color='skyblue', edgecolor='black')
    plt.title("", fontsize=12)  # simpler titles
    plt.xlabel(parameter_names[i], fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()  # adjust spacing automatically
plt.show()

# ---- Compute IQR for best parameter sets ----

# 25th percentile (Q1) and 75th percentile (Q3)
Q1 = np.percentile(best_params, 25, axis=0)
Q3 = np.percentile(best_params, 75, axis=0)

IQR = Q3 - Q1

print("\nInterquartile Ranges (IQR) for Best Parameter Sets:")
for name, q1, q3, iqr in zip(parameter_names, Q1, Q3, IQR):
    print(f"{name}:  Q1={q1:.16f},  Q3={q3:.16f},  IQR={iqr:.16f}")


end_time = time.time()

diff = end_time - start_time

print(diff)

