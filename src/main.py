"""
Author: Jake Vick
Purpose: End-to-end execution of distribution fitting, visualization, and risk metrics

Run w/ python -m src.main
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .distributions import fit_normal, fit_student_t
from .visualization import (
    plot_histogram_with_fits,
    plot_qq_plots,
    plot_tail_comparison
)
from .risk_metrics import (
    compute_var_normal,
    compute_cvar_normal,
    compute_var_student_t,
    compute_cvar_student_t
)

def main(alpha: float = 0.05):
    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / "data.csv"

    returns = pd.read_csv(returns_path)["return"]

    # Fit distributions
    normal_params = fit_normal(returns)
    t_params = fit_student_t(returns)

    # Visual diagnostics
    plot_histogram_with_fits(returns, normal_params, t_params, alpha)
    plot_qq_plots(returns, normal_params, t_params)
    plot_tail_comparison(normal_params, t_params)

    # Risk metrics
    mu, sigma = normal_params
    df, loc, scale = t_params

    print("\nRisk Metrics (95%)\n")

    print(f"Normal VaR:  {compute_var_normal(mu, sigma, alpha):.4f}")
    print(f"Normal CVaR: {compute_cvar_normal(mu, sigma, alpha):.4f}\n")

    print(f"Student-t VaR:  {compute_var_student_t(df, loc, scale, alpha):.4f}")
    print(f"Student-t CVaR: {compute_cvar_student_t(df, loc, scale, alpha):.4f}")

    plt.show()


if __name__ == "__main__":
    main()
