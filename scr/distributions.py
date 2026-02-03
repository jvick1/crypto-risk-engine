"""
Author: Jake Vick
Purpose: Compute Normal & Student-t distributions for VaR / CVaR modeling
"""

from typing import Tuple
from scipy.stats import norm, t

import numpy as np
import pandas as pd

def fit_normal(returns: pd.Series) -> Tuple[float, float]:
    """Fit a Normal distribution to returns using MLE.
    
    Parameters ----------
    returns: Series of asset returns.
    
    Output ----------
    mu, sigma: Mean and standard deviation.
    """

    mu = np.mean(returns)
    sigma = np.std(returns)
    return mu, sigma

def fit_student_t(returns: pd.Series) -> Tuple[float, float, float]:
    """Fit a Student's t distribution to returns using MLE.
    
    Parameters ----------
    returns: Series of asset returns.
    
    Output ----------
    df, loc, scale: Degrees of freedom (df), location (mean), scale (std).
    """

    df, loc, scale = t.fit(returns)
    return df, loc, scale