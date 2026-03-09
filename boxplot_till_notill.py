# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:09:16 2025

@author: Chinthaka
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Folders (adjust if needed) ---
folder_random1 = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\till_notill_random2"
folder_cluster1 = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\cluster_till_notill1"
folder_cluster2 = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\Cluster2_till_notill"
folder_polycluster = r"C:\Users\Chinthaka\OneDrive\Desktop\ABC\July\polycluster_till_notill"

# --- Load posterior TXT files (whitespace separated, no header) ---
posterior_files = {
    "Random ": pd.read_csv(
        os.path.join(folder_random1, "posterior_samples_cluster_till_notill_random_cluster_till_notill.txt"),
        header=None, sep=r"\s+"
    ),
    "Cluster 1": pd.read_csv(
        os.path.join(folder_cluster1, "posterior_samples_cluster2till_notill_August_8_27.txt"),
        header=None, sep=r"\s+"
    ),
    "Cluster 2": pd.read_csv(
        os.path.join(folder_cluster2, "posterior_samples_cluster2till_notill_August_8_28.txt"),
        header=None, sep=r"\s+"
    ),
    "Polycluster": pd.read_csv(
        os.path.join(folder_polycluster, "posterior_samples_polycluster_till_notill_cluster2_till_notill2.txt"),
        header=None, sep=r"\s+"
    ),
}

# Column order in our saved posterior files:
# theta, beta_non, xi, tau_non, rho_beta, rho_tau, d_threshold
params = ["theta","beta_non","xi","tau_non","rho_beta","rho_tau","d_threshold"]

for key in posterior_files:
    posterior_files[key].columns = params
    posterior_files[key]["Scenario"] = key

# Combine all
combined_df = pd.concat(posterior_files.values(), ignore_index=True)

# ===================== BOXPLOT FIGURE (math labels, single title, minimal x-labels) =====================
latex_param_labels = {
    "theta":        r"$\theta$",
    "beta_non":     r"$\beta_{\mathrm{non}}$",
    "xi":           r"$\xi$",
    "tau_non":      r"$\tau_{\mathrm{non}}$",
    "rho_beta":     r"$\rho_{\beta}$",
    "rho_tau":      r"$\rho_{\tau}$",
    "d_threshold":  r"$d$",
}

# 7 params -> use 4 rows x 2 cols (last panel empty & turned off)
nrows, ncols = 4, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 20), sharex=True)
axes = axes.flatten()

# font sizes
YLABEL_FONTSIZE = 24     # axis label (parameter name) on Y
YTICK_FONTSIZE  = 16     # numeric tick labels on Y
XLABEL_FONTSIZE = 24
XTICK_FONTSIZE  = 22
OFFSET_FONTSIZE = 14
# draw the 7 boxplots

custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b", "#e377c2"
]
for i, param in enumerate(params):
    ax = axes[i]

# Then use:
    sns.boxplot(data=combined_df,x="Scenario",y=param,ax=ax,palette=custom_colors,   # <--- change made here
        width=0.45
        )

    ax.set_ylabel(latex_param_labels.get(param, param), fontsize=YLABEL_FONTSIZE, labelpad=6)
    ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)  # <-- make Y ticks larger
    ax.set_title("")
    ax.set_xlabel("")  # we'll label only two axes later
    
    ax.yaxis.get_offset_text().set_fontsize(OFFSET_FONTSIZE)
    ax.yaxis.get_offset_text().set_fontweight("bold")


# turn off any unused axes
for j in range(len(params), nrows * ncols):
    axes[j].axis("off")

# figure out which two axes to show x tick labels on
num_panels = len(params)              # 7
last_used = num_panels - 1            # 6
last_row = last_used // ncols         # 3
col_in_last = last_used % ncols       # 1

idx_left = last_row * ncols           # bottom-left used axis
if col_in_last == ncols - 1:
    idx_right = last_row * ncols + (ncols - 1)  # full row: actual bottom-right
else:
    idx_right = last_row * ncols - 1 if last_row > 0 else last_used  # previous row's rightmost

label_axes = {idx_left, idx_right}

# hide x tick labels everywhere except the chosen two
for k, ax in enumerate(axes[:num_panels]):  # only the used axes
    if k in label_axes:
        ax.set_xlabel("Scenario", fontsize=XLABEL_FONTSIZE,fontweight="bold")
        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=XTICK_FONTSIZE)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
            tick.set_ha('center')
    else:
        ax.set_xlabel("")
        ax.tick_params(axis='x', which='both', labelbottom=False)


# room for larger y labels
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.subplots_adjust(left=0.10)  # increase if labels still clip

# optionally align y labels across columns
fig.align_ylabels(axes[:num_panels])

plt.show()

# ========================================================================================================

# ---------- Summary statistics: median and IQR ----------
def summarize_group(df, cols, scenario_label):
    out = {}
    for c in cols:
        q1 = df[c].quantile(0.25)
        q2 = df[c].quantile(0.50)
        q3 = df[c].quantile(0.75)
        out[c] = {"median": q2, "IQR": (q3 - q1)}
    return pd.DataFrame(out).T.assign(Scenario=scenario_label)

summaries = [summarize_group(df, params, scen) for scen, df in combined_df.groupby("Scenario")]
summary_df = pd.concat(summaries, axis=0).reset_index().rename(columns={"index": "Parameter"})

# Rounding map uses EXACT parameter names
round_map = {
    "theta": 3,          # small numbers: keep ~3 sig figs
    "beta_non": 3,
    "xi": 2,
    "tau_non": 3,
    "rho_beta": 3,
    "rho_tau": 3,
    "d_threshold": 2,
}

def _round_val(p, v):
    if p in ("theta", "beta_non"):
        return float(f"{float(v):.3g}")  # sig figs
    return round(float(v), round_map.get(p, 2))

summary_rounded = summary_df.copy()
summary_rounded["median"] = summary_rounded.apply(lambda r: _round_val(r["Parameter"], r["median"]), axis=1)
summary_rounded["IQR"]    = summary_rounded.apply(lambda r: _round_val(r["Parameter"], r["IQR"]), axis=1)

def format_entry(param, med, iqr):
    return f"{med} ({iqr})"

summary_rounded["Entry"] = summary_rounded.apply(
    lambda r: format_entry(r["Parameter"], r["median"], r["IQR"]), axis=1
)

# Pivot to Scenario × Parameter in the desired order
table_df = summary_rounded.pivot(index="Scenario", columns="Parameter", values="Entry")[
    ["theta","beta_non","xi","tau_non","rho_beta","rho_tau","d_threshold"]
].reset_index()

# Rename columns to LaTeX labels (consistent names)
latex_colmap = {
    "theta": r"$\theta$",
    "beta_non": r"$\beta_{\mathrm{non}}$",
    "xi": r"$\xi$",
    "tau_non": r"$\tau_{\mathrm{non}}$",
    "rho_beta": r"$\rho_{\beta}$",
    "rho_tau": r"$\rho_{\tau}$",
    "d_threshold": r"$d_{\mathrm{0}}$",
}
latex_df = table_df.rename(columns=latex_colmap)

# Wrap small numbers for siunitx \num{} in LaTeX
def wrap_siunitx(cell, wrap=False):
    if not wrap or not isinstance(cell, str):
        return cell
    try:
        med_str, iqr_str = cell.split(" (")
        iqr_str = iqr_str.rstrip(")")
        return r"\num{" + med_str + r"} (\num{" + iqr_str + r"})"
    except Exception:
        return cell

for col in [r"$\theta$", r"$\beta_{\mathrm{non}}$"]:
    latex_df[col] = latex_df[col].apply(lambda s: wrap_siunitx(s, wrap=True))

# Build LaTeX table
cols_order = [r"$\theta$", r"$\beta_{\mathrm{non}}$", r"$\xi$", r"$\tau_{\mathrm{non}}$", r"$\rho_{\beta}$", r"$\rho_{\tau}$", r"$d_{\mathrm{0}}$"]

latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Posterior medians with interquartile ranges (IQR) by scenario; entries are median (IQR).}")
latex_lines.append(r"\label{tab:posterior_summary_box}")
latex_lines.append(r"\begin{tabular}{lccccccc}")  # 1 (Scenario) + 7 params = 8 columns
latex_lines.append(r"\toprule")
latex_lines.append(r"\textbf{Scenario} & $\theta$ & $\beta_{\mathrm{non}}$ & $\xi$ & $\tau_{\mathrm{non}}$ & $\rho_{\beta}$ & $\rho_{\tau}$ & $d_{\mathrm{0}}$ \\")
latex_lines.append(r"\midrule")

for _, row in latex_df.iterrows():
    scenario = row["Scenario"]
    values = [row[c] for c in cols_order]
    latex_lines.append(scenario + " & " + " & ".join(values) + r" \\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_table_str = "\n".join(latex_lines)

os.makedirs("summaries_out", exist_ok=True)
summary_df.to_csv("summaries_out/posterior_summary_raw.csv", index=False)
summary_rounded.to_csv("summaries_out/posterior_summary_rounded.csv", index=False)

with open("summaries_out/posterior_summary_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table_str)

print("\n===== LaTeX table (copy into our .tex) =====\n")
print(latex_table_str)
print("\nSaved CSVs and LaTeX to: summaries_out/")
