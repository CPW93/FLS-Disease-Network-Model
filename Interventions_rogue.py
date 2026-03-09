# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:13:17 2025

@author: Chinthaka
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.neighbors import KDTree
from scipy.stats import sem

# ----------------- Reproducibility -----------------
np.random.seed(42)

# ----------------- Network Creation -----------------
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

def create_combined_distance_matrix(coordinates):
    all_points = np.vstack(coordinates)
    tree = KDTree(all_points)
    num_points = len(all_points)
    D = dok_matrix((num_points, num_points), dtype=np.float32)
    for i, p in enumerate(all_points):
        dists, idxs = tree.query([p], k=num_points)
        D[i, idxs[0]] = dists[0]
    return D.toarray(), all_points


def create_adjacency_matrix(distance_matrix, d_threshold):
    
    M = (distance_matrix > 0) & (distance_matrix <= d_threshold)
    np.fill_diagonal(M, 0)
    return csr_matrix(M.astype(np.int8))

def get_susceptible_to_infected_distances(D, S_idx, I_idx):
    # Robust shapes to avoid broadcasting errors
    if len(S_idx) == 0 and len(I_idx) == 0:
        return np.zeros((0, 0))
    if len(S_idx) == 0:
        return np.zeros((0, len(I_idx)))
    if len(I_idx) == 0:
        return np.zeros((len(S_idx), 0))
    return D[np.ix_(S_idx, I_idx)]

# ----------------- Fixed-Total Removal Schedule (bounded by end day) -----------------
def build_fixed_removal_schedule_bounded(start, end_inclusive, interval, total_to_remove):
    """
    Build {t: count} at t = start, start+interval, ..., <= end_inclusive.
    Distribute 'total_to_remove' evenly across events; remainder is front-loaded.
    """
    if total_to_remove <= 0:
        return {}
    event_times = list(range(start, end_inclusive + 1, interval))
    if not event_times:
        return {}
    n_events = len(event_times)
    base = total_to_remove // n_events
    rem  = total_to_remove %  n_events
    sched = {}
    for i, t in enumerate(event_times):
        sched[t] = base + (1 if i < rem else 0)
    return sched


# ---- Plot mapping (6 plots from create_graphs_with_gaps order) ----
def build_node_to_plot_map(coordinates):
    """
    coordinates: list of 6 lists; each list is the coords of one plot, in the order they were created.
    Returns:
        node_to_plot: array of length N giving the plot index (0..5) for each node in the vstack order
    """
    node_to_plot = []
    for p_idx, coords in enumerate(coordinates):
        node_to_plot.extend([p_idx] * len(coords))
    return np.array(node_to_plot, dtype=int)

# top row (row=0) -> plots 0,1,2 ; bottom row (row=1) -> plots 3,4,5
def build_plot_practice():
    """
    Return an array of length 6 with entries 'till' or 'no-till'.
    By your design: top row = tilled (0,1,2), bottom row = no-till (3,4,5).
    """
    return np.array(['till','till','till','no-till','no-till','no-till'], dtype=object)

def SEIRB_network_tillage_PeriodicRogue(
    G, A, timesteps,
    theta,                # plant-to-plant transmission hazard multiplier
    beta_non,             # soil-to-plant base for no-till
    rho_beta,             # till multiplier: beta_till = beta_non * rho_beta   (expect < 1)
    sigma, gamma,         # E->I and I->R daily hazards -> probs via 1-exp(-sigma), 1-exp(-gamma)
    xi,                   # infectious plants' contribution to soil
    r, k,                 # soil logistic growth parameters
    tau_non,              # soil removal for no-till
    rho_tau,              # till multiplier: tau_till = tau_non * rho_tau      (expect > 1)
    initial_infectednodes,
    combined_distance_matrix, d_threshold,   # (not used directly; A should already encode this)
    node_to_plot,         # array length N -> plot index (0..P-1)
    plot_practice,        # array length P -> 'till' or 'no-till'
    B0_non=4000.0,        # initial soil contamination for no-till plots
    B0_till=4000.0,       # initial soil contamination for tilled plots

    # ------- Roguing controls -------
    roguing_pct=0.0,                # if >0, remove this fraction of CURRENT infected on roguing days
    roguing_interval=None,          # e.g., every 7 days; used if removal_schedule is None
    roguing_start=0,                # first day roguing can happen
    strategy='random',              # 'random' or 'targeted' (frontier-based)
    removal_schedule=None           # dict: {timestep: quota}. If provided, overrides pct/interval logic
):
    """
    Returns
    -------
    status_matrix : (timesteps, 4) array
        Daily counts of S, E, I, R.
    B_history : (timesteps, P) array
        Per-plot soil contamination levels through time.
    removed_per_timestep : (timesteps,) int array
        Number of infected plants rogued (set to R) at each day.
    """
    N = G.number_of_nodes()
    P = int(np.max(node_to_plot) + 1)

    # --- Per-plot parameters from practice ---
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

    # --- States: 0=S, 1=E, 2=I, 3=R ---
    node_states = np.zeros(N, dtype=int)
    node_states[np.asarray(initial_infectednodes, dtype=int)] = 2

    status_matrix = np.zeros((timesteps, 4), dtype=float)
    B_history     = np.zeros((timesteps, P), dtype=float)
    removed_per_timestep = np.zeros(timesteps, dtype=int)

    # record t=0
    status_matrix[0] = [
        np.sum(node_states == 0),
        np.sum(node_states == 1),
        np.sum(node_states == 2),
        np.sum(node_states == 3),
    ]
    B_history[0] = Bp.copy()

    # carry-over for schedule quotas if not enough I on a given day
    carry_over = 0

    for t in range(1, timesteps):
        # ---------- Infection hazards (NETWORK + SOIL) ----------
        infected_mask   = (node_states == 2)
        infected_idx    = np.where(infected_mask)[0]
        susceptible_idx = np.where(node_states == 0)[0]

        # Local network risk (counts of infected neighbors within threshold)
        infected_neighbors = A @ infected_mask.astype(np.int8)    # length-N (int counts)
        local_risk = theta * infected_neighbors                   # hazard from network

        # Soil term varies by plot
        soil_term = beta_p[node_to_plot] * Bp[node_to_plot]       # length-N

        lambda_total = local_risk + soil_term                     # total hazard per node

        # Gate infections: S must have >=1 infected neighbor within threshold
        if susceptible_idx.size:
            has_close_infected = infected_neighbors[susceptible_idx] > 0
            if has_close_infected.any():
                p_inf = 1.0 - np.exp(-lambda_total[susceptible_idx])
                u = np.random.rand(susceptible_idx.size)
                newly_exposed_idx = susceptible_idx[(u < p_inf) & has_close_infected]
            else:
                newly_exposed_idx = np.array([], dtype=int)
        else:
            newly_exposed_idx = np.array([], dtype=int)

        # E -> I and I -> R (natural recovery) from hazards
        p_inc = 1.0 - np.exp(-sigma)
        p_rec = 1.0 - np.exp(-gamma)

        exposed_mask = (node_states == 1)
        to_infected  = exposed_mask & (np.random.rand(N) < p_inc)
        to_recovered = infected_mask & (np.random.rand(N) < p_rec)

        node_states[newly_exposed_idx] = 1
        node_states[to_infected]       = 2
        node_states[to_recovered]      = 3

        # ---------- Roguing step (AFTER natural transitions) ----------
        infected_idx = np.where(node_states == 2)[0]  # refresh
        num_to_recover = 0

        # How many to remove today
        if removal_schedule is not None:
            quota_today = removal_schedule.get(t, 0)
            quota = carry_over + quota_today
            if quota > 0 and infected_idx.size > 0:
                num_to_recover = min(quota, infected_idx.size)
                carry_over = quota - num_to_recover
            else:
                # no infected or no quota today → carry forward any remaining quota
                carry_over = quota
        else:
            if (roguing_pct > 0) and (roguing_interval is not None):
                if (t >= roguing_start) and ((t - roguing_start) % roguing_interval == 0):
                    if infected_idx.size > 0:
                        num_to_recover = int(np.floor(roguing_pct * infected_idx.size))

        # Which infected to remove
        if num_to_recover > 0 and infected_idx.size > 0:
            k = min(num_to_recover, infected_idx.size)

            if strategy == 'targeted':
                # prioritize infected with the largest current I→S frontier
                sus_mask = (node_states == 0)
                if sus_mask.any():
                    # rows: current infected; cols: current susceptibles (already thresholded by A)
                    I_to_S = A[infected_idx][:, sus_mask]                      # sparse submatrix
                    frontier = np.asarray(I_to_S.sum(axis=1)).ravel()          # # susceptible neighbors per infected

                    if frontier.max() > 0:
                        order = np.argsort(-frontier)[:k]                      # largest frontier first
                        to_recover = infected_idx[order]
                    else:
                        # no susceptible neighbors adjacent → fall back to random
                        to_recover = np.random.choice(infected_idx, size=k, replace=False)

                else:
                    # no susceptibles left → removing any infected is equivalent
                    to_recover = np.random.choice(infected_idx, size=k, replace=False)
            else:
                # random strategy
                to_recover = np.random.choice(infected_idx, size=k, replace=False)

            node_states[to_recover] = 3
            removed_per_timestep[t] = k
        else:
            removed_per_timestep[t] = 0

        # ---------- Soil update (per-plot) AFTER roguing ----------
        infected_mask = (node_states == 2)
        I_counts = np.bincount(node_to_plot[infected_mask], minlength=P)  # per-plot I

        growth       = r * Bp * (1.0 - Bp / k)
        decay        = tau_p * Bp
        contribution = xi * I_counts
        Bp = np.maximum(Bp + growth - decay + contribution, 0.0)

        # ---------- Record ----------
        status_matrix[t] = [
            np.sum(node_states == 0),
            np.sum(node_states == 1),
            np.sum(node_states == 2),
            np.sum(node_states == 3),
        ]
        B_history[t] = Bp

    return status_matrix, B_history, removed_per_timestep


# ---------- summary table (mean ± SD) ----------

def summarize_for_table(results, removed_series_list, dt=1.0, N_total=None, T_total=None):
    """
    results: list of runs; each run is array T x 4 [S,E,I,R]
    removed_series_list: list of arrays (length T) of removed counts per timestep
    dt: time step (days). Your sim uses daily steps → dt=1.0
    N_total, T_total: if provided, also return normalized AUCs (per plant-day)
    """
    arr = np.array(results)  # (reps, T, 4)
    S = arr[:, :, 0]
    I = arr[:, :, 2]

    # Per-run metrics
    peak_values = I.max(axis=1)          # peak infected per run
    peak_times  = I.argmax(axis=1)       # time of peak per run
    final_I     = I[:, -1]               # infected at final timestep
    final_S     = S[:, -1]               # healthy at final timestep

    # AUDPC/AUC for infected (trapezoidal rule)
    
    auc_I       = np.trapz(I, dx=dt, axis=1)          # absolute AUC (infected*days)
    auc_I_norm  = None
    if (N_total is not None) and (T_total is not None) and (N_total > 0) and (T_total > 0):
        # fraction of the maximum possible infected*days (= N_total * T_total)
        auc_I_norm = auc_I / (N_total * (T_total * dt))

    removed_tot = np.array([rs.sum() for rs in removed_series_list])

    out = {
        "peak_values": peak_values,
        "peak_times":  peak_times,
        "final_I":     final_I,
        "final_S":     final_S,
        "removed_tot": removed_tot,
        "auc_I":       auc_I
    }
    if auc_I_norm is not None:
        out["auc_I_norm"] = auc_I_norm
    return out


def plot_early_late_side_by_side(results_early, results_late, T, title_left, title_right, savepath=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left plot
    ax = axes[0]
    for label, results in results_early.items():
        I = np.array(results)[:, :, 2]
        mean = I.mean(axis=0)
        ci = 1.96 * sem(I, axis=0)
        ax.plot(mean, label=label)
        ax.fill_between(range(T), mean - ci, mean + ci, alpha=0.2)
    ax.set_title(title_left); ax.set_xlabel("Timesteps (Days)"); ax.set_ylabel("Infected Plants")
    ax.grid(True); ax.legend()

    # Right plot
    ax = axes[1]
    for label, results in results_late.items():
        I = np.array(results)[:, :, 2]
        mean = I.mean(axis=0); ci = 1.96 * sem(I, axis=0)
        ax.plot(mean, label=label)
        ax.fill_between(range(T), mean - ci, mean + ci, alpha=0.2)
    ax.set_title(title_right); ax.set_xlabel("Timesteps (Days)")
    ax.grid(True); ax.legend()

    fig.suptitle("Early vs Late Roguing (mean ± 95% CI)", y=1.03, fontsize=12)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


# ----------------- Params -----------------
params = dict(
    theta = 0.0001,
    beta_non = 0.00000078,
    rho_beta = 0.9,
    sigma=0.1,
    gamma=1/75,
    xi=1500,
    tau_non=0.01,
    rho_tau = 1.2,
    r=0.001,
    k=60000,
    d_threshold= 30
)

T = 140 # total number of timesteps
reps = 15

# ----------------- Build Network -----------------
dimensions = [(72, 4)] * 6
gap = [150, 150]
x_scale, y_scale = 8.33, 76.2
coordinates = create_graphs_with_gaps(dimensions, gap, x_scale, y_scale)
Dmat, all_coords = create_combined_distance_matrix(coordinates)
A = create_adjacency_matrix(Dmat, d_threshold=30)
G = nx.from_scipy_sparse_array(A)

node_to_plot = build_node_to_plot_map(coordinates)
plot_practice = build_plot_practice()


# Load fixed initial infected nodes
initial_infectednodes = np.loadtxt(
    r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\till_notill_random2\initial_infected_nodes_cluster_till_notill_random.txt",
    dtype=int
).tolist()

# ----------------- percentage and schedule settings -----------------
N = G.number_of_nodes()          # 1728 is number of total plants 
pct_remove = 0.02               # removal percentage like 11.6% from total plants
TOTAL_REMOVE = int(round(pct_remove * N))   # ≈ 200

# Biweekly cadence, stop at day 126 (no removals on day 140)
EARLY_START, LATE_START = 35, 42 
END_DAY = 138
INTERVAL = 1

early_sched = build_fixed_removal_schedule_bounded(EARLY_START, END_DAY, INTERVAL, TOTAL_REMOVE)
late_sched  = build_fixed_removal_schedule_bounded(LATE_START,  END_DAY, INTERVAL, TOTAL_REMOVE)

# ----------------- Run: No Roguing baseline -----------------
res_no_shared, rm_no_shared = [], []
for _ in range(reps):
    s0, _, rm0 = SEIRB_network_tillage_PeriodicRogue(
        G, A.copy(), T, **params,
        initial_infectednodes=initial_infectednodes,
        combined_distance_matrix=Dmat,
        roguing_pct=0.0, roguing_interval=None, removal_schedule=None, 
        node_to_plot = build_node_to_plot_map(coordinates),
        plot_practice = build_plot_practice(),
        strategy='random'
    )
    res_no_shared.append(s0); rm_no_shared.append(rm0)

# ----------------- Run: Early and Late, Random vs Targeted -----------------

""" res_early_rand = will store the full simulation result matrices 
(the status_matrix arrays returned from SEIR_Spreadfunction_PeriodicRogue) 
for the early random roguing scenario, one entry per replicate.

rm_early_rand = will store the removed-per-timestep arrays for those same runs 
— this tracks how many infected plants were removed at each timestep. """

# ---------- run all scenarios for one interval ----------
def cadence_label(dt):
        return "Daily (Δt=1)" if dt == 1 else ("Every 3 days (Δt=3)" if dt == 3 else (f"Weekly (Δt={dt})"))

def simulate_early_late_for_interval(interval, reps, total_remove,
                                     EARLY_START=35, LATE_START=42, END_DAY=138):
    early_sched = build_fixed_removal_schedule_bounded(EARLY_START, END_DAY, interval, total_remove)
    late_sched  = build_fixed_removal_schedule_bounded(LATE_START,  END_DAY, interval, total_remove)

    res_early_rand, rm_early_rand = [], []
    res_early_targ, rm_early_targ = [], []
    res_late_rand,  rm_late_rand  = [], []
    res_late_targ,  rm_late_targ  = [], []

    for _ in range(reps):
        # Early
        se_r, _, rme_r = SEIRB_network_tillage_PeriodicRogue(
            G, A.copy(), T, **params,
            initial_infectednodes=initial_infectednodes,
            combined_distance_matrix=Dmat,
            removal_schedule=early_sched,
            node_to_plot=node_to_plot, plot_practice=plot_practice,
            strategy='random'
        )
        se_t, _, rme_t = SEIRB_network_tillage_PeriodicRogue(
            G, A.copy(), T, **params,
            initial_infectednodes=initial_infectednodes,
            combined_distance_matrix=Dmat,
            removal_schedule=early_sched,
            node_to_plot=node_to_plot, plot_practice=plot_practice,
            strategy='targeted'
        )
        # Late
        sl_r, _, rml_r = SEIRB_network_tillage_PeriodicRogue(
            G, A.copy(), T, **params,
            initial_infectednodes=initial_infectednodes,
            combined_distance_matrix=Dmat,
            removal_schedule=late_sched,
            node_to_plot=node_to_plot, plot_practice=plot_practice,
            strategy='random'
        )
        sl_t, _, rml_t = SEIRB_network_tillage_PeriodicRogue(
            G, A.copy(), T, **params,
            initial_infectednodes=initial_infectednodes,
            combined_distance_matrix=Dmat,
            removal_schedule=late_sched,
            node_to_plot=node_to_plot, plot_practice=plot_practice,
            strategy='targeted'
        )

        res_early_rand.append(se_r); rm_early_rand.append(rme_r)
        res_early_targ.append(se_t); rm_early_targ.append(rme_t)
        res_late_rand.append(sl_r);  rm_late_rand.append(rml_r)
        res_late_targ.append(sl_t);  rm_late_targ.append(rml_t)

    results_early = {"No Roguing": res_no_shared,
                     "Random Roguing": res_early_rand,
                     "Targeted Roguing": res_early_targ}
    results_late  = {"No Roguing": res_no_shared,
                     "Random Roguing": res_late_rand,
                     "Targeted Roguing": res_late_targ}
    return results_early, results_late

# ---------- 3×2 grid plot across intervals ----------
def plot_early_late_grid_by_intervals(intervals, reps, total_remove,
                                      EARLY_START=35, LATE_START=42, END_DAY=138,
                                      savepath=None, return_cache=False):
    COLORS = {"No Roguing": "C0", "Random Roguing": "C1", "Targeted Roguing": "C2"}
    STYLES = {"No Roguing": dict(linestyle="-", marker=None),
              "Random Roguing": dict(linestyle="--", marker=None),
              "Targeted Roguing": dict(linestyle="-", marker="X",
                                       markerfacecolor="white", markeredgewidth=1.8)}
    MARK_EVERY, LINEWIDTH, MARKERSIZE = 6, 2, 6

    nrows = len(intervals)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5*nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    letter_idx = 0

    # --- cache to reuse elsewhere ---
    cache = {}  # keys: (interval, "early"|"late") -> results dict

    for i, interval in enumerate(intervals):
        results_early, results_late = simulate_early_late_for_interval(
            interval, reps, total_remove, EARLY_START, LATE_START, END_DAY
        )
        cache[(interval, "early")] = results_early
        cache[(interval, "late")]  = results_late

        # ----- LEFT (EARLY) -----
        axL = axes[i, 0]
        for label, results in results_early.items():
            I = np.array(results)[:, :, 2]
            mean, ci = I.mean(axis=0), 1.96 * sem(I, axis=0)
            (line,) = axL.plot(mean, label=label, color=COLORS.get(label),
                               linewidth=LINEWIDTH, **STYLES.get(label, {}),
                               markevery=MARK_EVERY, markersize=MARKERSIZE)
            axL.fill_between(range(len(mean)), mean-ci, mean+ci, alpha=0.2, color=line.get_color())
        if i == 0:
            axL.set_title(f"EARLY (start {EARLY_START})")
        axL.set_ylabel("Infected Plants"); axL.grid(True)
        axL.text(0.015, 0.96, letters[letter_idx], transform=axL.transAxes,
                 fontweight='bold', fontsize=18, va='top', ha='left'); letter_idx += 1
        axL.text(-0.12, 0.5, cadence_label(interval), transform=axL.transAxes,
                 rotation=90, va='center', ha='right', fontsize=16)
        axL.tick_params(labelbottom=(i == nrows - 1))
        if i == nrows - 1:
            axL.set_xlabel("Timesteps (Days)", fontsize=16)

        # ----- RIGHT (LATE) -----
        axR = axes[i, 1]
        for label, results in results_late.items():
            I = np.array(results)[:, :, 2]
            mean, ci = I.mean(axis=0), 1.96 * sem(I, axis=0)
            (line,) = axR.plot(mean, label=label, color=COLORS.get(label),
                               linewidth=LINEWIDTH, **STYLES.get(label, {}),
                               markevery=MARK_EVERY, markersize=MARKERSIZE)
            axR.fill_between(range(len(mean)), mean-ci, mean+ci, alpha=0.2, color=line.get_color())
        if i == 0:
            axR.set_title(f"LATE (start {LATE_START})")
        axR.grid(True)
        axR.text(0.015, 0.96, letters[letter_idx], transform=axR.transAxes,
                 fontweight='bold', fontsize=18, va='top', ha='left'); letter_idx += 1
        axR.tick_params(labelbottom=(i == nrows - 1))
        if i == nrows - 1:
            axR.set_xlabel("Timesteps (Days)", fontsize=16)
            axR.legend(loc="lower right",fontsize=20)

    fig.suptitle("Early vs Late Roguing (mean ± 95% CI) across Intervals", y=1, fontsize=22)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight"); print(f"Saved: {savepath}")
    plt.show()

    if return_cache:
        return cache

def plot_daily_vs_weekly_from_cache(cache, EARLY_START=35, LATE_START=42, title=None, savepath=None):
    COLORS = {"No Roguing": "C0", "Random Roguing": "C1", "Targeted Roguing": "C2"}
    STYLES = {"No Roguing": dict(linestyle="-", marker=None),
              "Random Roguing": dict(linestyle="--", marker=None),
              "Targeted Roguing": dict(linestyle="-", marker="X",
                                       markerfacecolor="white", markeredgewidth=1.8)}
    LINEWIDTH, MARKERSIZE, MARK_EVERY = 2, 6, 6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    # Left = (a): Δt=1, EARLY
    res_early_daily = cache[(1, "early")]
    axL = axes[0]
    for label, results in res_early_daily.items():
        I = np.array(results)[:, :, 2]
        mean, ci = I.mean(axis=0), 1.96 * sem(I, axis=0)
        t = np.arange(len(mean))
        (line,) = axL.plot(mean, label=label, color=COLORS.get(label),
                           linewidth=LINEWIDTH, **STYLES.get(label, {}),
                           markevery=MARK_EVERY, markersize=MARKERSIZE)
        axL.fill_between(t, mean-ci, mean+ci, alpha=0.2, color=line.get_color())
    axL.set_title(f"Daily Roguing (Δt=1, Early start {EARLY_START})")
    axL.set_ylabel("Infected Plants"); axL.grid(True, alpha=0.4)

    # Right = (f): Δt=7, LATE
    res_late_weekly = cache[(7, "late")]
    axR = axes[1]
    for label, results in res_late_weekly.items():
        I = np.array(results)[:, :, 2]
        mean, ci = I.mean(axis=0), 1.96 * sem(I, axis=0)
        t = np.arange(len(mean))
        (line,) = axR.plot(mean, label=label, color=COLORS.get(label),
                           linewidth=LINEWIDTH, **STYLES.get(label, {}),
                           markevery=MARK_EVERY, markersize=MARKERSIZE)
        axR.fill_between(t, mean-ci, mean+ci, alpha=0.2, color=line.get_color())
    axR.set_title(f"Weekly Roguing (Δt=7, Late start {LATE_START})", fontsize=18)
    axR.grid(True, alpha=0.4); axR.legend(loc="lower right")

    for ax in axes:
        ax.set_xlabel("Timesteps (Days)")
    if title is None:
        title = "Daily Roguing vs Weekly Roguing (mean ± 95% CI)"
    fig.suptitle(title, y=1.02, fontsize=18)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight"); print(f"Saved: {savepath}")
    plt.show()


cache = plot_early_late_grid_by_intervals(
    intervals=[1,3,7],
    reps=reps,
    total_remove=TOTAL_REMOVE,
    EARLY_START=EARLY_START,
    LATE_START=LATE_START,
    END_DAY=END_DAY,
    savepath="rogue_grid_1x3x2.png",
    return_cache=True
)

plot_daily_vs_weekly_from_cache(cache,
    EARLY_START=EARLY_START, LATE_START=LATE_START,
    savepath="rogue_daily_vs_weekly.png")


# ----------------- Core summary table ( (a) and (f) requested columns) -----------------

def summarize_from_runs(runs, dt=1.0):
    arr = np.array(runs)               # shape: (reps, T, 4)
    S = arr[:, :, 0]; I = arr[:, :, 2]
    peak_vals = I.max(axis=1)
    peak_times = I.argmax(axis=1)
    final_I = I[:, -1]
    final_S = S[:, -1]
    return dict(
        peak_mu=int(round(peak_vals.mean())),  peak_sd=int(round(peak_vals.std(ddof=1))),
        tpeak_mu=int(round(peak_times.mean())), tpeak_sd=int(round(peak_times.std(ddof=1))),
        finI_mu=int(round(final_I.mean())),     finI_sd=int(round(final_I.std(ddof=1))),
        finS_mu=int(round(final_S.mean())),     finS_sd=int(round(final_S.std(ddof=1))),
    )

def build_core_summary_from_cache(cache, intervals, rho_display, EARLY_START=35, LATE_START=42):
    rows = []
    for dt in intervals:
        resE = cache[(dt, "early")]
        resL = cache[(dt, "late")]
        def add_block(start_day, timing_label, resdict):
            for label in ["No Roguing", "Random Roguing", "Targeted Roguing"]:
                stats = summarize_from_runs(resdict[label])
                rows.append({
                    "Start": start_day,
                    "Timing": timing_label,
                    "Roguing Strategy": label,
                    "Δt": dt,
                    "ρ": rho_display,
                    "Peak": f"{stats['peak_mu']} ± {stats['peak_sd']}",
                    "Peak Time(days)": f"{stats['tpeak_mu']} ± {stats['tpeak_sd']}",
                    "I(T=138)": f"{stats['finI_mu']} ± {stats['finI_sd']}",
                    "Healthy": f"{stats['finS_mu']} ± {stats['finS_sd']}",
                })
        add_block(EARLY_START, "Early", resE)
        add_block(LATE_START,  "Late",  resL)

    df = pd.DataFrame(rows)
    order = {"No Roguing":0, "Random Roguing":1, "Targeted Roguing":2}
    df["STRAT_ORD"] = df["Roguing Strategy"].map(order)
    df = df.sort_values(["Timing","Δt","STRAT_ORD"]).drop(columns="STRAT_ORD").reset_index(drop=True)
    return df

# build directly from the cache you already created for the figure
core_df_cache = build_core_summary_from_cache(
    cache, intervals=[1,3,7], rho_display=pct_remove,
    EARLY_START=EARLY_START, LATE_START=LATE_START
)

# LaTeX (same header/format as before)
latex_header_names = {
    "Start": r"\textbf{Start}",
    "Timing": r"\textbf{Timing}",
    "Roguing Strategy": r"\textbf{Roguing Strategy}",
    "Δt": r"$\boldsymbol{\Delta t}$",
    "ρ": r"$\boldsymbol{\rho}$",
    "Peak": r"\textbf{Peak}",
    "Peak Time(days)": r"\textbf{Peak Time(days)}",
    "I(T=138)": r"\textbf{$I(T{=}138)$}",
    "Healthy": r"\textbf{Healthy}",
}
latex_df = core_df_cache.rename(columns=latex_header_names)
latex_str = latex_df.to_latex(index=False, escape=False,
    column_format="c c l c c c c c c",
    caption=(f"Absolute outcomes (mean $\\pm$ SD) by timing, strategy, and cadence "
             f"(total removals = {TOTAL_REMOVE}, start$_\\text{{early}}$={EARLY_START}, "
             f"start$_\\text{{late}}$={LATE_START})."),
    label="tab:rogue_core")
with open("rogue_core_summary_from_cache.tex","w") as f:
    f.write(latex_str)
print(latex_str[:800])

