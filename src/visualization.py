"""
Author: Jake Vick
Purpose: Visualization of Normal & Student-t distributions 
"""

from scipy.stats import norm, t, probplot
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_histogram_with_fits(returns, normal_params, t_params, alpha=0.05):
    """
    Plot histogram w/ fitted Normal and Student-t prob density functions (PDFs)
    
    Parameters ----------
    returns: Our log returns
    normal_params: normal dist
    t_params: student t dist
    alpha: tail prob for VaR overlay

    Output ----------
    PDFs histogram
    """
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
    """
    Generates quantile-quantile (QQ) plots comparing returns to fitted Normal and Student-t. Good for normal dist.
    
    Parameters ----------
    returns: Our log returns
    normal_params: normal dist
    t_params: student t dist

    Output ----------
    Two QQ-plots 
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    probplot(returns, dist=norm(*normal_params), plot=axs[0])
    axs[0].set_title("QQ Plot vs Normal")

    probplot(returns, dist=t(*t_params), plot=axs[1])
    axs[1].set_title("QQ Plot vs Student-t")

    return fig


def plot_tail_comparison(normal_params, t_params, tail_min=-0.30):
    """
    Plot tail compairson for Normal and Student-t on a log scale - useful when extreme events are common
    
    Parameters ----------
    normal_params: normal dist
    t_params: student t dist
    tail_min: min for tail viz

    Output ----------
    left-tail density comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_tail = np.linspace(tail_min, 0, 1000)
    ax.plot(x_tail, norm.pdf(x_tail, *normal_params), label="Normal", color="red")
    ax.plot(x_tail, t.pdf(x_tail, *t_params), label="Student-t", color="green")

    ax.set_yscale("log")
    ax.set_title("Left-Tail Density Comparison (Log Scale)")
    ax.legend()
    return fig

def plot_rolling_var(results_df: pd.DataFrame,
                     summary_df: pd.DataFrame,
                     methods: list = ["normal", "student_t", "historical"],
                     alpha: float = 0.05,
                     coin_symbol: str = None):

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actual returns
    ax.plot(results_df["date"],
            results_df["actual_return"],
            color="black",
            alpha=0.5,
            linewidth=1,
            label="Actual Return")

    # Color map for consistency
    colors = {
        "normal": "#1f77b4",
        "student_t": "#ff7f0e",
        "historical": "#2ca02c"
    }

    # Plot VaR lines + breaches
    for method in methods:
        breach_rate = summary_df.loc[method, "breach_rate"]

        # VaR line
        ax.plot(results_df["date"],
                results_df[f"{method}_var"],
                linestyle="--",
                linewidth=2,
                color=colors.get(method, None),
                label=f"{method.capitalize()} VaR ({breach_rate:.2%})")

        # Breach markers
        breach_mask = results_df[f"{method}_breach"]
        ax.scatter(
            results_df.loc[breach_mask, "date"],
            results_df.loc[breach_mask, "actual_return"],
            marker="x",
            s=40,
            color=colors.get(method, None),
            alpha=0.9
        )

    # Zero return line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # Formatting
    title_symbol = coin_symbol.upper() if coin_symbol else ""
    ax.set_title(f"Rolling {int(alpha*100)}% VaR Backtest {title_symbol}",
                 fontsize=14,
                 fontweight="bold")

    ax.set_ylabel("Log Returns")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    """
    When you run it as a module, make sure you are in the \student_t_var_cvar\ folder and run `python -m src.visualization`, Python understands the package structure and relative imports correctly.
    """
    from .distributions import fit_normal, fit_student_t
    
    parser = argparse.ArgumentParser(description="Viz for Normal & Student-t distributions")
    parser.add_argument(
        "--coin_symbol",
        type=str,
        default="btc",
        help="Coin symbol to load returns for (e.g., btc, eth, sol)"
    )
    parser.add_argument(
        "--vs_currency",
        type=str,
        default="usd",
        help="Quote currency (default: usd)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"

    try:
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