
import os
import networkx as nx
import random
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from itertools import product
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.stats._distn_infrastructure import rv_frozen, rv_sample
from scipy.stats import rv_discrete, rv_continuous
from scipy.spatial import KDTree
from scipy.sparse import dok_matrix
from sklearn.neighbors import NearestNeighbors

"""Create Network"""
###############################################################################

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
        node_to_plot: array of length N giving the plot index (0,1,2,3,4,5) for each node in the vstack order,
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
    theta,               # plant-to-plant (theta)
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

def compute_summaries(status_matrix):
    """ record infections as dataframe """
   
    dictSums = dict()
    dictSums['S'] = status_matrix[:, 0].tolist()  # Susceptible
    dictSums['E'] = status_matrix[:, 1].tolist()  # Exposed
    dictSums['I'] = status_matrix[:, 2].tolist()  # Infected
    dictSums['R'] = status_matrix[:, 3].tolist()  # Recovered
    resDicts = dictSums

    return resDicts['I']

def _merge_dict(dict1, dict2):
    """
    Function to merge two dictionaries
    """
    res = {**dict1, **dict2}
    return res

def data_indiv_simulation(model, prior_args_model=None, fixed_args_model=None):
    if prior_args_model is None:
        prior_args_model = dict()

    if fixed_args_model is None:
        fixed_args_model = dict()

    dict_params = dict()
    for key, value in prior_args_model.items():
        try:
            if isinstance(value, (rv_frozen, rv_sample, rv_discrete)):
                sample = value.rvs(random_state=ss.randint(0, 4294967296).rvs())
                dict_params[key] = sample
                print(f" Sampled param: {key} = {sample} (type: {type(sample)})")
            elif isinstance(value, (list, np.ndarray)):
                sample = np.random.choice(value)
                dict_params[key] = sample
                print(f" Chose from list: {key} = {sample}")
            else:
                raise ValueError(f"[ERROR] Invalid prior for key '{key}': {value} (type {type(value)})")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to sample key: {key}, prior value: {value}\nException: {e}")

    args_model = _merge_dict(dict_params, fixed_args_model)
    print(" Final args_model['d_threshold'] =", args_model.get("d_threshold"), type(args_model.get("d_threshold")))
    
    dict_summaries = model(**args_model)
    
    return dict_summaries, dict_params

##################################################################################    

# Define the distance that we will use, the standardized Euclidean or Absolute distance
def distance_abs(df_sim_summaries, df_obs_summaries):   

    dist = np.sqrt(np.sum(np.array( ( (df_sim_summaries.iloc[0,:] - df_obs_summaries.iloc[0,:]) )**2 )))   
    return dist

def distance_func(df_sim_summaries, df_obs_summaries):  
      return distance_abs(df_sim_summaries, df_obs_summaries)

def _is_discrete(dist):
    "To identify whether the ss.rv_frozen dist is discrete"
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else:
        return isinstance(dist, rv_discrete)

def _is_continuous(dist):
    "To identify whether the ss.rv_frozen dist is continuous"
   
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_continuous)
    else:
        return isinstance(dist, rv_continuous)


def _perturb_discrete_param_on_support(prior_disc, perturb_kernel):
    """ Perturb a discrete parameter thanks to a truncated Gaussian distribution,
    and rounding of the perturbed value.
   
    We use a perturbation distribution (perturb_kernel) centered at the value
    to perturb (which is discrete). The perturbed parameter value is then
    rounded to fall in beans of size 1, centered at the integer value.If the
    perturbed parameter value falls in the support of the discrete prior
    distribution for the parameter then we accept this value, otherwise we keep
    simulating to fall in the support.
   
    """
   
    # Generate the perturbed value
    perturbed_float = perturb_kernel.rvs()
    perturbed_int = np.round(perturbed_float)
    while perturbed_int < prior_disc.support()[0] or perturbed_int > prior_disc.support()[1]:
        perturbed_float = perturb_kernel.rvs()
        perturbed_int = np.round(perturbed_float)        

    return perturbed_int

def _perturb_continuous_param_on_support(prior_cont, perturb_kernel):
    """ Perturb a continuous parameter thanks to a truncated Gaussian distribution """
   
    # Generate the perturbed value
    perturbed_float = perturb_kernel.rvs()
    while perturbed_float < prior_cont.support()[0] or perturbed_float > prior_cont.support()[1]:
        perturbed_float = perturb_kernel.rvs()

    return perturbed_float

def run_grid_search(initial_infectednodes, label="scenario"):
    """
    Fast grid search using LOG-SPACED grids for positive params (no zeros),
    thinner levels, same return signature as original function.
    """
    fixed_args_model = {
        "timesteps": 139,             
        "sigma": 1/10,
        "gamma": 1/75,
        "r": 0.001,
        "k": 60000,
        "node_to_plot": node_to_plot,
        "plot_practice": plot_practice,
        "initial_infectednodes": initial_infectednodes,
    }

    # ---- thin, sensible ranges (log-space for positive tiny params) ----
    
    theta_range    = np.logspace(-6, -3, 5)      # 5 pts: 1e-6..1e-3
    beta_non_range = np.logspace(-9, -5, 5)     # 5 pts: 1e-10..1e-5
    xi_range       = np.logspace(1, 3.8, 5)    # (5 pts: ~ 500..6300 
    tau_non_range  = np.logspace(-3, -1, 4)      # 4 pts: 1e-3..1e-1

    rho_beta_range = [0.1, 1.0, 1.5, 2]             
    rho_tau_range  = [0.1, 1.0, 1.5, 2]             
    dthresh_range  = [10, 70, 100, 130, 160, 190, 220, 250, 280, 310]

    # Size: 5*5*5*4*4*4*10 = 80000 sims, much faster.
    grid = list(product(theta_range, beta_non_range, rho_beta_range,
                   xi_range, dthresh_range, tau_non_range, rho_tau_range))

    rows = []
    print(f"[{label.upper()}] Starting LOG-thin grid over {np.prod([len(theta_range),len(beta_non_range),len(rho_beta_range),len(xi_range),len(dthresh_range),len(tau_non_range),len(rho_tau_range)])} combos...")
    
    for (theta, beta_non, rho_beta, xi, d_threshold, tau_non, rho_tau) in tqdm(grid):
        sim_args = {
            'theta': theta, 'beta_non': beta_non, 'rho_beta': rho_beta,
            'xi': xi, 'd_threshold': d_threshold, 'tau_non': tau_non, 'rho_tau': rho_tau,
            **fixed_args_model
        }
        try:
            sim_output = SEIR_model(**sim_args)
            sim_summary = compute_summaries(sim_output)
            df_sim_summaries = pd.DataFrame([sim_summary])[df_obs_summaries.columns]
            dist = distance_abs(df_sim_summaries, df_obs_summaries)
        except Exception as e:
            print(f"Simulation failed at {sim_args}: {e}")
            dist = np.inf
        rows.append({**sim_args, 'distance': dist})

    df_grid_results = pd.DataFrame(rows).sort_values(by='distance').reset_index(drop=True)

    # Save full results and best-100
    grid_results_path = os.path.join(folder_path, f"grid_search_results_{label}.csv")
    top100_results_path = os.path.join(folder_path, f"top_100_grid_results_{label}.csv")
    df_grid_results.to_csv(grid_results_path, index=False)
    top_params = df_grid_results.head(100)
    top_params.to_csv(top100_results_path, index=False)

    # Real-scale bounds from winners (same as before)
    theta_bounds     = (top_params['theta'].min(),        top_params['theta'].max())
    beta_non_bounds  = (top_params['beta_non'].min(),     top_params['beta_non'].max())
    rho_beta_bounds  = (top_params['rho_beta'].min(),     top_params['rho_beta'].max())
    xi_bounds        = (top_params['xi'].min(),           top_params['xi'].max())
    dthresh_bounds   = (top_params['d_threshold'].min(),  top_params['d_threshold'].max())
    tau_non_bounds   = (top_params['tau_non'].min(),      top_params['tau_non'].max())
    rho_tau_bounds   = (top_params['rho_tau'].min(),      top_params['rho_tau'].max())

    # --- Build priors for ABC ---
    # log-normal for positive tiny params (stabilizes ABC); uniform for the rest.
    def _fit_lognorm(x):
        x = np.asarray(x, float)
        x = x[x > 0]
        if x.size == 0:
            return ss.lognorm(s=1.0, scale=np.exp(np.log(1e-6)))
        ln = np.log(np.clip(x, 1e-300, None))
        mu = float(np.mean(ln))
        sd = float(np.std(ln, ddof=1)) if ln.size > 1 else 0.3
        return ss.lognorm(s=sd, scale=np.exp(mu))

    prior_args_model = {
        "theta":       _fit_lognorm(top_params['theta'].values),
        "beta_non":    _fit_lognorm(top_params['beta_non'].values),
        "xi":          _fit_lognorm(top_params['xi'].values),
        "tau_non":     _fit_lognorm(top_params['tau_non'].values),
        "rho_beta":    ss.uniform(loc=rho_beta_bounds[0], scale=rho_beta_bounds[1]-rho_beta_bounds[0]),
        "rho_tau":     ss.uniform(loc=rho_tau_bounds[0],  scale=rho_tau_bounds[1]-rho_tau_bounds[0]),
        "d_threshold": ss.uniform(loc=dthresh_bounds[0],  scale=dthresh_bounds[1]-dthresh_bounds[0]),
    }

    print(f"[{label.upper()}] Best parameter ranges:")
    print("theta      :", theta_bounds)
    print("beta_non   :", beta_non_bounds)
    print("rho_beta   :", rho_beta_bounds)
    print("xi         :", xi_bounds)
    print("d_threshold:", dthresh_bounds)
    print("tau_non    :", tau_non_bounds)
    print("rho_tau    :", rho_tau_bounds)

    return top_params, prior_args_model, theta_bounds, beta_non_bounds, rho_beta_bounds, xi_bounds, dthresh_bounds, tau_non_bounds, rho_tau_bounds


###############################################################################

# Observed data and their corresponding time steps
observed_timesteps = [0, 45, 50, 75, 89, 96, 117, 138]

# Data points to store
infected_nodes = [52, 86, 104, 138, 276, 362, 484, 622]
observed_values = infected_nodes

# Create a DataFrame with a single row
df_obs_summaries = pd.DataFrame([infected_nodes])
df_obs_summaries.columns = observed_timesteps

# Print the DataFrame
print(df_obs_summaries)


dimensions = [(72, 4), (72, 4),(72, 4), (72, 4), (72, 4), (72, 4)]  # Example dimensions for the grids
gaps = [150, 150] # gap between plots
x_scale = 8.33 # gap between two plants in a row
y_scale = 76.2 # gap between two rows

coordinates = create_graphs_with_gaps(dimensions, gaps, x_scale, y_scale)
combined_distance_matrix = create_combined_distance_matrix(coordinates)

combined_adjacency_matrix = create_adjacency_matrix(combined_distance_matrix)

node_to_plot = build_node_to_plot_map(coordinates)
plot_practice = build_plot_practice()

G = nx.from_numpy_array(combined_adjacency_matrix)
A = nx.to_scipy_sparse_array(G)

# Define fixed parameters

fixed_args_model = {
    "timesteps": 139,
    "sigma": 1 / 10,
    "gamma": 1 / 75,
    "r": 0.001,
    "k": 60000,
    "node_to_plot": node_to_plot,
    "plot_practice": plot_practice
}

# ABC-SMC configuration
threshold_init = 20000
threshold_final = 50
alpha = 0.1
scale_factor = 2
num_acc_sim = 200
folder_path = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July"
os.makedirs(folder_path, exist_ok=True)

#---------------------------------------------------------------------------------------------------------------------------

""" Implementation of the replenishment SMC ABC algorithm.

We here implement the replenishment SMC ABC algorithm, proposed by Drovandi
and Pettitt, (2011).

Drovandi, C. C. and Pettitt, A. N. "Estimation of Parameters for
Macroparasite Population Evolution Using Approximate Bayesian Computation"
Biometrics, 67, 225-233, (2011)."""

def abc_RSMCABC(model,
                prior_args_model=None, fixed_args_model=None,
                threshold_init=threshold_init, threshold_final=threshold_final,
                alpha=alpha, scale_factor=scale_factor,
                perturbation="Gaussian",
                num_acc_sim=num_acc_sim, df_observed_summaries=None,
                distance_func=distance_func):
    """ Implementation of the replenishment SMC ABC algorithm. """
    # Recover a list for the parameter priors
    list_priors = []
    disc_identifier = []
    # For each key, we recover from the prior
    for (key, value) in prior_args_model.items():
        list_priors += [value]
        disc_identifier += [_is_discrete(value)]
   
    # Uniform distribution for MH test
    unif_dist = ss.uniform(loc = 0, scale = 1)
   
    # Number of particles to discard at each step
    num_drop_sim = int(alpha * num_acc_sim)
    if num_drop_sim == 0:
        num_drop_sim = 1
   
    # Identify the summary statistics to keep while simulating
    cols_to_keep = df_observed_summaries.columns
   
    step_count = 0      # number of sequential steps
    sim_count_total = 0 # total number of simulated data
   
    # To store accepted weights/parameters values, and distances
    df_params = pd.DataFrame()
    df_dist_acc = pd.DataFrame()
   
    # Keep track of the epsilon values
    epsilon_values = [threshold_init]
   
    if step_count == 0:
       
        sim_count = 0 # number of accepted simulations during the current step
       
        ### Initial classic rejection sampling algorithm
        
        while sim_count < num_acc_sim:
            sim_count_total += 1
    
            if sim_count_total % 10 == 0:  # Print every 10 simulations
                print(f"Simulations tried: {sim_count_total}, Accepted: {sim_count}")
    
            start_sim = time.time()
            
            # Simulate parameters
            sim_values_args_model = dict()
            for key, value in prior_args_model.items():
                if isinstance(value, (rv_frozen, rv_sample, rv_discrete)):
                    sample = value.rvs()
                    sim_values_args_model[key] = sample
                    print(f" Sampled param: {key} = {sample} (type: {type(sample)})")
                elif isinstance(value, (list, np.ndarray)):
                    sample = np.random.choice(value)
                    sim_values_args_model[key] = sample
                    print(f" Chose from list: {key} = {sample}")
                else:
                    raise ValueError(f"[ERROR] Invalid prior for key '{key}': {value} (type {type(value)})")

            
            # Merge fixed parameters
            args_model = _merge_dict(sim_values_args_model, fixed_args_model)
    
    # Simulate from model
            data_sim = model(**args_model)  

            sim_time = time.time() - start_sim
            print(f"  Simulation took: {sim_time:.2f} sec")
    
    # Compute summary statistics
            start_summary = time.time()
            dict_summaries = compute_summaries(data_sim)
            summary_time = time.time() - start_summary
            print(f"  Summary computation took: {summary_time:.2f} sec")
    
    # Compute distance
            start_dist = time.time()
            df_summaries = pd.DataFrame([dict_summaries])
            df_summaries_reduced = df_summaries[df_observed_summaries.columns]
            dist = distance_func(df_summaries_reduced, df_observed_summaries)
            dist_time = time.time() - start_dist
            
            print(df_summaries_reduced)
            print(dist)
            print(f"  Distance computation took: {dist_time:.2f} sec")
    
    # Check acceptance
            if dist <= threshold_init:
                df_params = pd.concat([df_params, pd.DataFrame([sim_values_args_model])], ignore_index=True)
                df_dist_acc = pd.concat([df_dist_acc, pd.DataFrame([dist])], ignore_index=True)
                sim_count += 1
       
        step_count += 1
   
    # SMC-ABC core part
    if step_count > 0:
       
        # Determine the order of the distances when sorted in increasing order
        idx_sort = np.argsort(df_dist_acc.iloc[:,0])
       
        # Reorder the parameters and distance with this order
        df_dist_acc = df_dist_acc.iloc[idx_sort,:]
        df_dist_acc = df_dist_acc.reset_index(drop=True)
               
        df_params = df_params.iloc[idx_sort,:]
        df_params = df_params.reset_index(drop=True)
               
        # Compute epsilon_max = the maximal distance
        epsilon_max = df_dist_acc.iloc[num_acc_sim-1,0]
       
        epsilon_values = epsilon_values + [epsilon_max]
       
        # while epsilon_max is greater than threshold_final
        while epsilon_max > threshold_final:
           
            print(epsilon_max, threshold_final)
           
            # Drop the num_drop_sim (Na) particles with largest distances
            df_params.drop(df_dist_acc.tail(num_drop_sim).index, inplace=True)
            df_dist_acc.drop(df_dist_acc.tail(num_drop_sim).index, inplace=True)
           
            epsilon_next = df_dist_acc.tail(1).iloc[0,0] # the largest distance of the remaining simulations
                       
            std_params = scale_factor * df_params.apply(np.std)
           
            ### Resample num_drop_sim new particles and data that are accepted
           
            num_acc_next = 0
           
            while num_acc_next < num_drop_sim:
           
                ### Sample an old parameter value from the
                ### num_acc_sim - num_drop_sim previously accepted values
                idx_sel = np.random.choice(df_params.index[:(num_acc_sim-num_drop_sim)])
                sim_count_total += 1
               
                ### Perturb the selected parameter values with a kernel
                             
                # Parameter perturbation
                prev_params = np.array(df_params.iloc[idx_sel,:])
               
                perturbed_params = np.empty(len(prev_params))
                # For each parameter value
                for i in range(len(prev_params)):
                    perturbation_kernel_Gauss = ss.norm(prev_params[i], std_params[i])
                    # if the parameter is discrete, we use a discretized Gaussian on the support of the prior
                    if disc_identifier[i]:
                        perturbed_params[i] = _perturb_discrete_param_on_support(list_priors[i], perturbation_kernel_Gauss)
                    # else we use a continuous Gaussian on the support of the prior
                    else:
                        perturbed_params[i] = _perturb_continuous_param_on_support(list_priors[i], perturbation_kernel_Gauss)
                   
                # To use the simulated parameters in our data generation function
                # we need a list of dict, with same structure as sim_args_model
                perturbed_params_dict = dict()
                idx_params = 0
                for (key, value) in sim_values_args_model.items():
                    # if the parameter is discrete, we need an integer for the mechanisms
                    if disc_identifier[idx_params]:
                        perturbed_params_dict[key] = int(perturbed_params[idx_params])
                    else:
                        perturbed_params_dict[key] = perturbed_params[idx_params]
                    idx_params += 1
               
                ### Generate a new data given the perturbed parameters
                args_model = _merge_dict(perturbed_params_dict,
                                         fixed_args_model)

                data_sim = model(**args_model)
                dict_summaries = compute_summaries(data_sim)
               
               
                df_summaries = pd.DataFrame([dict_summaries])
               
               
                df_summaries_reduced = df_summaries[cols_to_keep]
               
   
                dist_new = distance_func(df_summaries_reduced,
                                         df_observed_summaries)

                if dist_new <= epsilon_next:
                   
                    print("Dist_new: ", dist_new, " Epsilon next: ", epsilon_next)
                                                           
                    # For the parameters
                    list_prior_params_old = []
                    list_prior_params_new = []
                    list_pdf_new_given_old = []
                    list_pdf_old_given_new = []
                    for i in range(len(list_priors)):
                        if disc_identifier[i]:
                            list_prior_params_old += [list_priors[i].pmf(prev_params[i])]
                            list_prior_params_new += [list_priors[i].pmf(perturbed_params[i])]  
                            list_pdf_new_given_old += [1 if std_params[i] == 0
                                                       else ss.norm(prev_params[i], std_params[i]).cdf(perturbed_params[i]+0.5) - ss.norm(prev_params[i], std_params[i]).cdf(perturbed_params[i]-0.5)]
                            list_pdf_old_given_new += [1 if std_params[i] == 0
                                                       else ss.norm(perturbed_params[i], std_params[i]).cdf(prev_params[i]+0.5) - ss.norm(perturbed_params[i], std_params[i]).cdf(prev_params[i]-0.5)]
                        else:
                            list_prior_params_old += [list_priors[i].pdf(prev_params[i])]
                            list_prior_params_new += [list_priors[i].pdf(perturbed_params[i])]
                            list_pdf_new_given_old += [ss.norm(prev_params[i], std_params[i]).pdf(perturbed_params[i])]
                            list_pdf_old_given_new += [ss.norm(perturbed_params[i], std_params[i]).pdf(prev_params[i])]

                    prior_ratio_params = np.prod(list_prior_params_new) / np.prod(list_prior_params_old)
                    transition_ratio_params = np.prod(list_pdf_old_given_new) / np.prod(list_pdf_new_given_old)
                   
                    mh_ratio = np.min([1, prior_ratio_params * transition_ratio_params])
                   
                    if unif_dist.rvs() < mh_ratio:
                           
                        if len(perturbed_params) > 0:
                            perturbed_params_df = pd.DataFrame(perturbed_params.reshape(-1, len(perturbed_params)),columns=df_params.columns)
                        
                            df_params = pd.concat([df_params,perturbed_params_df], ignore_index=True)
                        else:
                            df_params = pd.concat([df_params, pd.DataFrame([], index=[1])], ignore_index=True)
                        df_dist_acc = pd.concat([df_dist_acc, pd.DataFrame([dist_new])], ignore_index=True)
                       
                        num_acc_next += 1
                       
            # Determine the order of the distances when sorted in increasing order
            idx_sort = np.argsort(df_dist_acc.iloc[:,0])
           
            # Reorder the parameters and distance with this order
            df_dist_acc = df_dist_acc.iloc[idx_sort,:]
            df_dist_acc = df_dist_acc.reset_index(drop=True)
                       
            df_params = df_params.iloc[idx_sort,:]
            df_params = df_params.reset_index(drop=True)
           
            # Compute epsilon_max = the maximal distance
            epsilon_max = df_dist_acc.iloc[num_acc_sim-1,0]
           
            epsilon_values = epsilon_values + [epsilon_max]
           
            step_count += 1
           
        threshold_values = np.array(epsilon_values)

        return df_params, df_dist_acc, sim_count_total, threshold_values
   
# -----------------------------
# Define Seed Scenarios
# -----------------------------

flat_coordinates = np.vstack(coordinates)

# Random seed selector
def get_initial_seeds(count=52, cluster_center=None):
    # ignore cluster_center if we want random seeds
    indices = np.random.choice(len(flat_coordinates), size=count, replace=False)
    return indices.tolist()

# -----------------------------
# Run ABC for both scenarios
# -----------------------------
scenarios = ["cluster_till_notill_random_dist_euclidean"]
posteriors = {}
results = {}
seed_locations = {}

for scenario in scenarios:
    print(f"\n--- Running scenario: {scenario.upper()} ---")
    seeds = get_initial_seeds(52) # always need to change

    # Save for later
    seed_locations[scenario] = seeds
    np.savetxt(os.path.join(folder_path, f"initial_infected_nodes_{scenario}.txt"), seeds, fmt='%d')

    _, prior_args_model, theta_bounds, beta_non_bounds, rho_beta_bounds, xi_bounds, dthresh_bounds, tau_non_bounds, rho_tau_bounds  = run_grid_search(
        seeds, label=scenario
    )

    fixed_args_model['initial_infectednodes'] = seeds

    #  Run ABC for this scenario
    df_params, df_dist, sim_count, thresholds = abc_RSMCABC(
        model=SEIR_model,
        prior_args_model=prior_args_model,
        fixed_args_model=fixed_args_model,
        threshold_init=threshold_init,
        threshold_final=threshold_final,
        alpha=alpha,
        scale_factor=scale_factor,
        perturbation="Gaussian",
        num_acc_sim=num_acc_sim,
        df_observed_summaries=df_obs_summaries,
        distance_func=distance_func
    )

    # Simulate a single run using the inferred priors for visualization
    data_summaries, dict_params = data_indiv_simulation(
        model=SEIR_model,
        prior_args_model=prior_args_model,
        fixed_args_model=fixed_args_model
    )
    results[scenario] = compute_summaries(data_summaries)

    # Save posterior
    posterior = df_params.values
    file_name = f"posterior_samples_{scenario}_cluster_till_notill.txt"
    file_path = os.path.join(folder_path, file_name)
    np.savetxt(file_path, posterior)
    print(f"Posterior for {scenario} saved at: {file_path}")






