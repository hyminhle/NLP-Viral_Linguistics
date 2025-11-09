"""

This module implements the Linguistic Contagion Model (LCM) based on 
the SIR (Susceptible-Infected-Recovered) epidemiological framework.

Equations (from proposal):
    dS/dt = -βSI - μS + γR  (Equation 1)
    dI/dt = βSI - αI - νI    (Equation 2)
    dR/dt = αI - γR          (Equation 3)
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SIRParameters:
    """Parameters for the SIR model"""
    beta: float = 0.3      # Transmission rate
    alpha: float = 0.1     # Recovery rate
    nu: float = 0.05       # Mutation rate
    mu: float = 0.02       # Birth rate (new users)
    gamma: float = 0.01    # Re-susceptibility rate
    

class LinguisticContagionModel:
    """
    Linguistic Contagion Model (LCM) based on SIR framework.
    
    Attributes:
        params (SIRParameters): Model parameters
        N (int): Total population size
        history (pd.DataFrame): Time series of S, I, R compartments
    """
    
    def __init__(self, params: Optional[SIRParameters] = None):
        """
        Initialize the Linguistic Contagion Model.
        
        Args:
            params: SIR parameters. If None, uses default values.
        """
        self.params = params if params else SIRParameters()
        self.N = None
        self.history = None
        self.fitted_params = None
        
    def sir_equations(self, y: np.ndarray, t: float, N: int, 
                     beta: float, alpha: float, nu: float, 
                     mu: float, gamma: float) -> List[float]:
        """
        SIR differential equations from the proposal.
        
        Args:
            y: Current state [S, I, R]
            t: Current time
            N: Total population
            beta, alpha, nu, mu, gamma: Model parameters
            
        Returns:
            List of derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        
        # Equation 1: dS/dt = -βSI - μS + γR
        dS = -beta * S * I / N - mu * S + gamma * R
        
        # Equation 2: dI/dt = βSI - αI - νI
        dI = beta * S * I / N - alpha * I - nu * I
        
        # Equation 3: dR/dt = αI - γR
        dR = alpha * I - gamma * R
        
        return [dS, dI, dR]
    
    def simulate(self, I0: int, N: int, t_max: int = 100, 
                 dt: float = 1.0) -> pd.DataFrame:
        """
        Simulate SIR model forward in time.
        
        Args:
            I0: Initial number of infected (users using the term)
            N: Total population size
            t_max: Maximum time steps
            dt: Time step size (days)
            
        Returns:
            DataFrame with columns: t, S, I, R, total
        """
        self.N = N
        
        # Initial conditions
        S0 = N - I0
        R0 = 0
        y0 = [S0, I0, R0]
        
        # Time points
        t = np.linspace(0, t_max, int(t_max / dt))
        
        # Solve ODE
        solution = odeint(
            self.sir_equations, 
            y0, 
            t, 
            args=(N, self.params.beta, self.params.alpha, 
                  self.params.nu, self.params.mu, self.params.gamma)
        )
        
        # Store results
        self.history = pd.DataFrame({
            't': t,
            'S': solution[:, 0],
            'I': solution[:, 1],
            'R': solution[:, 2],
            'total': solution.sum(axis=1)
        })
        
        return self.history
    
    def calculate_R0(self) -> float:
        """
        Calculate basic reproduction number R0.
        
        R0 = β / (α + ν)
        
        If R0 > 1, the linguistic innovation will spread (go viral)
        If R0 < 1, it will die out
        
        Returns:
            R0 value
        """
        return self.params.beta / (self.params.alpha + self.params.nu)
    
    def find_peak(self) -> Tuple[float, float]:
        """
        Find peak infection time and magnitude.
        
        Returns:
            Tuple of (peak_time, peak_infections)
        """
        if self.history is None:
            raise ValueError("Must run simulate() first")
        
        peak_idx = self.history['I'].idxmax()
        peak_time = self.history.loc[peak_idx, 't']
        peak_infections = self.history.loc[peak_idx, 'I']
        
        return peak_time, peak_infections
    
    def fit_to_data(self, time_series: pd.Series, population: int,
                   bounds: Optional[Dict] = None) -> SIRParameters:
        """
        Fit SIR parameters to observed time series data using optimization.
        
        Args:
            time_series: Pandas Series with datetime index and infection counts
            population: Total population size
            bounds: Optional dict with parameter bounds
            
        Returns:
            Fitted SIRParameters object
        """
        # Prepare data
        t_data = np.arange(len(time_series))
        I_data = time_series.values
        
        # Initial guess
        I0 = I_data[0]
        S0 = population - I0
        R0 = 0
        y0 = [S0, I0, R0]
        
        # Default bounds
        if bounds is None:
            bounds = {
                'beta': (0.01, 1.0),
                'alpha': (0.01, 0.5),
                'nu': (0.0, 0.2),
                'mu': (0.0, 0.1),
                'gamma': (0.0, 0.1)
            }
        
        def objective(params):
            """Minimize squared error between model and data"""
            beta, alpha, nu, mu, gamma = params
            
            solution = odeint(
                self.sir_equations,
                y0,
                t_data,
                args=(population, beta, alpha, nu, mu, gamma)
            )
            
            I_pred = solution[:, 1]
            return np.sum((I_pred - I_data) ** 2)
        
        # Initial parameter guess
        x0 = [0.3, 0.1, 0.05, 0.02, 0.01]
        
        # Parameter bounds as list of tuples
        param_bounds = [bounds[p] for p in ['beta', 'alpha', 'nu', 'mu', 'gamma']]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=param_bounds)
        
        # Store fitted parameters
        self.fitted_params = SIRParameters(
            beta=result.x[0],
            alpha=result.x[1],
            nu=result.x[2],
            mu=result.x[3],
            gamma=result.x[4]
        )
        
        # Update current parameters
        self.params = self.fitted_params
        
        return self.fitted_params
    
    def predict(self, t_future: int) -> pd.DataFrame:
        """
        Predict future dynamics beyond training data.
        
        Args:
            t_future: Number of time steps to predict into the future
            
        Returns:
            DataFrame with predictions
        """
        if self.history is None:
            raise ValueError("Must run simulate() first")
        
        # Get final state from history
        S_final = self.history['S'].iloc[-1]
        I_final = self.history['I'].iloc[-1]
        R_final = self.history['R'].iloc[-1]
        y0 = [S_final, I_final, R_final]
        
        # Time points for prediction
        t_start = self.history['t'].iloc[-1]
        t = np.linspace(t_start, t_start + t_future, t_future)
        
        # Solve ODE
        solution = odeint(
            self.sir_equations,
            y0,
            t,
            args=(self.N, self.params.beta, self.params.alpha,
                  self.params.nu, self.params.mu, self.params.gamma)
        )
        
        # Format predictions
        predictions = pd.DataFrame({
            't': t,
            'S': solution[:, 0],
            'I': solution[:, 1],
            'R': solution[:, 2]
        })
        
        return predictions
    
    def plot_dynamics(self, actual_data: Optional[pd.Series] = None,
                     title: str = "SIR Model Dynamics",
                     figsize: Tuple[int, int] = (12, 6)):
        """
        Plot SIR dynamics over time.
        
        Args:
            actual_data: Optional actual infection counts to overlay
            title: Plot title
            figsize: Figure size
        """
        if self.history is None:
            raise ValueError("Must run simulate() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: All compartments
        ax1.plot(self.history['t'], self.history['S'], 'b-', 
                label='Susceptible (S)', linewidth=2)
        ax1.plot(self.history['t'], self.history['I'], 'r-', 
                label='Infected (I)', linewidth=2)
        ax1.plot(self.history['t'], self.history['R'], 'g-', 
                label='Recovered (R)', linewidth=2)
        
        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Number of Users', fontsize=12)
        ax1.set_title(f'{title}\nR₀ = {self.calculate_R0():.2f}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model vs Actual
        ax2.plot(self.history['t'], self.history['I'], 'r-', 
                label='SIR Model', linewidth=2)
        
        if actual_data is not None:
            t_actual = np.arange(len(actual_data))
            ax2.plot(t_actual, actual_data, 'ko-', 
                    label='Actual Data', markersize=4, alpha=0.6)
            
            # Calculate RMSE
            min_len = min(len(actual_data), len(self.history))
            rmse = np.sqrt(np.mean((self.history['I'][:min_len] - actual_data[:min_len]) ** 2))
            ax2.text(0.05, 0.95, f'RMSE: {rmse:.2f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Time (days)', fontsize=12)
        ax2.set_ylabel('Active Users', fontsize=12)
        ax2.set_title('Model Fit', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def summary_statistics(self) -> Dict:
        """
        Calculate summary statistics for the model.
        
        Returns:
            Dictionary with key statistics
        """
        if self.history is None:
            raise ValueError("Must run simulate() first")
        
        peak_time, peak_infections = self.find_peak()
        final_recovered = self.history['R'].iloc[-1]
        
        stats = {
            'R0': self.calculate_R0(),
            'peak_time': peak_time,
            'peak_infections': peak_infections,
            'total_infected_pct': (final_recovered / self.N) * 100,
            'final_susceptible': self.history['S'].iloc[-1],
            'final_recovered': final_recovered,
            'params': self.params.__dict__
        }
        
        return stats


def compare_scenarios(scenarios: Dict[str, SIRParameters], 
                     I0: int, N: int, t_max: int = 100) -> pd.DataFrame:
    """
    Compare multiple parameter scenarios.
    
    Args:
        scenarios: Dict mapping scenario names to SIRParameters
        I0: Initial infections
        N: Population size
        t_max: Simulation time
        
    Returns:
        DataFrame with results for each scenario
    """
    results = {}
    
    for name, params in scenarios.items():
        model = LinguisticContagionModel(params)
        model.simulate(I0, N, t_max)
        
        results[name] = {
            'R0': model.calculate_R0(),
            'peak_time': model.find_peak()[0],
            'peak_infections': model.find_peak()[1],
            'final_infected_pct': (model.history['R'].iloc[-1] / N) * 100
        }
    
    return pd.DataFrame(results).T


if __name__ == "__main__":
    # Example usage
    print("Viral Linguistics: SIR Model Framework")
    print("=" * 50)
    
    # Initialize model with default parameters
    model = LinguisticContagionModel()
    
    # Simulate
    N = 10000  # Total population
    I0 = 10    # Initial adopters
    history = model.simulate(I0, N, t_max=100)
    
    # Calculate statistics
    stats = model.summary_statistics()
    
    print(f"\nModel Statistics:")
    print(f"  R₀: {stats['R0']:.2f}")
    print(f"  Peak Time: Day {stats['peak_time']:.1f}")
    print(f"  Peak Infections: {stats['peak_infections']:.0f}")
    print(f"  Total Infected: {stats['total_infected_pct']:.1f}%")
    
    if stats['R0'] > 1:
        print("\n✓ This term will GO VIRAL (R₀ > 1)")
    else:
        print("\n✗ This term will NOT spread (R₀ < 1)")
    
    # Plot
    fig = model.plot_dynamics(title="Example: Linguistic Innovation Spread")
    plt.show()