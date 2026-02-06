"""
Author: Jake Vick
Purpose: Visualization of Normal & Student-t distributions 
"""

from scipy.stats import norm, t, probplot
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_histogram_with_fits(returns, normal_params, t_params, alpha=0.05):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(returns, kde=False, stat="density", ax=ax, label="Empirical")

    x = np.linspace(returns.min(), returns.max(), 1000)
    ax.plot(x, norm.pdf(x, *normal_params), label="Normal Fit", color="red")
    ax.plot(x, t.pdf(x, *t_params), label="Student-t Fit", color="green")

    var_n = normal_params[0] + normal_params[1] * norm.ppf(alpha)
    var_t = t_params[1] + t_params[2] * t.ppf(alpha, t_params[0])

    ax.axvline(var_n, color="red", linestyle="--", label="Normal VaR")
    ax.axvline(var_t, color="green", linestyle="--", label="Student-t VaR")

    ax.set_title("Return Distribution with Fitted PDFs")
    ax.legend()
    return fig


def plot_qq_plots(returns, normal_params, t_params):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    probplot(returns, dist=norm(*normal_params), plot=axs[0])
    axs[0].set_title("QQ Plot vs Normal")

    probplot(returns, dist=t(*t_params), plot=axs[1])
    axs[1].set_title("QQ Plot vs Student-t")

    return fig


def plot_tail_comparison(normal_params, t_params, tail_min=-0.10):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_tail = np.linspace(tail_min, 0, 1000)
    ax.plot(x_tail, norm.pdf(x_tail, *normal_params), label="Normal", color="red")
    ax.plot(x_tail, t.pdf(x_tail, *t_params), label="Student-t", color="green")

    ax.set_yscale("log")
    ax.set_title("Left-Tail Density Comparison (Log Scale)")
    ax.legend()
    return fig


if __name__ == "__main__":
    from .distributions import fit_normal, fit_student_t
    
    try:
        base_dir = Path(__file__).resolve().parents[1]
        returns_path = base_dir / "data" / "output" / "data.csv"

        returns = pd.read_csv(returns_path)["return"]

        # Fit models (for testing only)
        normal_params = fit_normal(returns)
        t_params = fit_student_t(returns)

        print("Visualization test:")
        print(f"Normal params: {normal_params}")
        print(f"Student-t params: {t_params}")

        # Generate plots
        plot_histogram_with_fits(returns, normal_params, t_params)
        plot_qq_plots(returns, normal_params, t_params)
        plot_tail_comparison(normal_params, t_params)

        plt.show()

    except Exception as e:
        print(f"Visualization test failed: {e}")