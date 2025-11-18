"""
Genome Classes for Regime-Based Evolution Testing
===============================================

Exact implementation of genome classes as specified by ChatGPT review:
- StaticGenome: fixed α
- PredictionGenome: α based on prediction error
- NeighborGenome: α based on neighbor performance  
- RewardGenomeV2: α based on reward EMA vs target

These genomes will be tested across 4 regimes to see which strategies dominate
in different environmental conditions.
"""

import math
import numpy as np


# ---------------------------------------------------------------------
# Base helper: safe sigmoid + clipping for floats
# ---------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Scalar sigmoid; assume x is a Python float or NumPy scalar."""
    return 1.0 / (1.0 + math.exp(-x))


def _clip(x: float, x_min: float, x_max: float) -> float:
    return max(x_min, min(x, x_max))


# ---------------------------------------------------------------------
# 1. StaticGenome – fixed alpha, never changes
# ---------------------------------------------------------------------

class StaticGenome:
    """
    Fixed learning rate genome.
    - alpha_W is set once and never changed.
    
    Should dominate in Regime 0 (Stable & Clean) where consistent 
    learning without over-tuning is optimal.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)

    def init_cell_state(self, cell):
        """
        Called once when the cell is created / reset.
        """
        cell.alpha_W = float(self.alpha)
        # keep alpha and alpha_W in sync for convenience
        cell.alpha = float(self.alpha)

    def update_learning_rate(self, cell, *args, **kwargs):
        """
        No-op: keeps alpha_W constant. Included for interface symmetry.
        """
        # just ensure both attributes exist
        cell.alpha_W = float(self.alpha)
        cell.alpha = float(self.alpha)
        return cell.alpha_W


# ---------------------------------------------------------------------
# 2. PredictionGenome – alpha based on prediction error
# ---------------------------------------------------------------------

class PredictionGenome:
    """
    Learning rate adapts based on prediction error vs an EMA of error.

    Behavior (Philosophy A):
    - If current error > EMA(error): increase alpha (struggling → learn faster)
    - If current error < EMA(error): decrease alpha (doing well → refine slowly)
    - Alpha is bounded in [alpha_min, alpha_max].
    
    Should dominate in Regime 1 (Slowly Drifting) where continuous 
    adaptation to changing dynamics is required.
    """

    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-2,
        gamma_error: float = 0.9,   # EMA smoothing of error
        sensitivity: float = 3.0,   # how strongly error vs EMA affects alpha
    ):
        assert 0.0 < alpha_min <= alpha_max
        assert 0.0 < gamma_error < 1.0

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.gamma_error = float(gamma_error)
        self.sensitivity = float(sensitivity)

    def init_cell_state(self, cell):
        """
        Initialize error EMA and starting alpha.
        """
        cell.error_ema = 0.0  # start with zero; will warm up
        mid = 0.5 * (self.alpha_min + self.alpha_max)
        cell.alpha_W = mid
        cell.alpha = mid

    def update_learning_rate(self, cell, prediction_error: float):
        """
        Update alpha_W based on current prediction error.

        Args:
            prediction_error: scalar (e.g. L2 norm of prediction error vector).
        """
        e_t = float(prediction_error)
        e_prev = float(getattr(cell, "error_ema", 0.0))
        e_ema = self.gamma_error * e_prev + (1.0 - self.gamma_error) * e_t
        cell.error_ema = e_ema

        # If e_t > e_ema → struggling → want alpha closer to alpha_max
        # If e_t < e_ema → doing well → want alpha closer to alpha_min
        # Use sigmoid on normalized difference
        if e_ema > 1e-12:
            diff = (e_t - e_ema) / (e_ema + 1e-12)
        else:
            diff = 0.0

        # Positive diff → increase lambda → alpha up
        lam = _sigmoid(self.sensitivity * diff)  # in (0,1)

        alpha_span = self.alpha_max - self.alpha_min
        alpha_t = self.alpha_min + lam * alpha_span
        alpha_t = _clip(alpha_t, self.alpha_min, self.alpha_max)

        cell.alpha_W = alpha_t
        cell.alpha = alpha_t

        return alpha_t


# ---------------------------------------------------------------------
# 3. NeighborGenome – alpha based on neighbor performance
# ---------------------------------------------------------------------

class NeighborGenome:
    """
    Learning rate adapts based on neighbor performance (social learning).

    Intended use:
    - Environment computes neighbor_avg_reward for each cell and calls
      update_from_network_signals() each step.

    Behavior idea:
    - If neighbors outperform the cell → increase alpha (learn more aggressively).
    - If neighbors underperform → decrease alpha (stabilize & trust own policy).
    
    Should dominate in Regime 3 (Cooperative) where neighbor coordination
    provides reward bonuses.
    """

    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-2,
        neighbor_weight: float = 0.5,   # scales how strongly neighbor diff affects alpha
    ):
        assert 0.0 < alpha_min <= alpha_max
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.neighbor_weight = float(neighbor_weight)

    def init_cell_state(self, cell):
        """
        Initialize alpha at midpoint.
        """
        mid = 0.5 * (self.alpha_min + self.alpha_max)
        cell.alpha_W = mid
        cell.alpha = mid

    def update_from_network_signals(
        self,
        cell,
        global_avg_reward: float,
        neighbor_avg_reward: float,
        step: int,
    ):
        """
        Called by the environment each step with network-level info.

        Args:
            cell: PopulationCell.core (NGISCell) or wrapper exposing average_reward()
            global_avg_reward: mean reward across all cells
            neighbor_avg_reward: mean reward of this cell's neighbors
            step: current global step (can be used for annealing if desired)
        """
        # self performance over recent horizon (env should define average_reward)
        if hasattr(cell, "average_reward"):
            self_perf = float(cell.average_reward())
        else:
            # fallback: use last_reward if that's all you have
            self_perf = float(getattr(cell, "last_reward", 0.0))

        # If neighbors are doing better → positive delta → increase alpha
        # If neighbors are doing worse → negative delta → decrease alpha
        delta = neighbor_avg_reward - self_perf

        # squash delta to [-1, 1]
        scaled = math.tanh(self.neighbor_weight * delta)

        span = self.alpha_max - self.alpha_min
        mid = 0.5 * (self.alpha_min + self.alpha_max)

        alpha_t = mid + 0.5 * span * scaled
        alpha_t = _clip(alpha_t, self.alpha_min, self.alpha_max)

        cell.alpha_W = alpha_t
        cell.alpha = alpha_t

        return alpha_t


# ---------------------------------------------------------------------
# 4. RewardGenomeV2 – alpha based on reward EMA vs target
# ---------------------------------------------------------------------

class RewardGenomeV2:
    """
    Reward-sensitive learning rate genome (stable, non-collapsing).

    Behavior (Philosophy A – verified):
    - reward > target_reward → decrease alpha (conservative when doing well)
    - reward < target_reward → increase alpha (aggressive when struggling)
    - Uses EMA of reward instead of raw instantaneous reward.
    - Alpha is always bounded in [alpha_min, alpha_max].
    
    Should dominate in Regime 2 (Abrupt Switches + High Noise) where
    rapid adaptation to performance drops is critical.
    """

    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-2,
        gamma_reward: float = 0.9,   # EMA factor for reward
        target_reward: float = 0.5,
        sensitivity: float = 5.0,    # k: how sharply adv affects alpha
    ):
        assert 0.0 < alpha_min <= alpha_max
        assert 0.0 < gamma_reward < 1.0

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.gamma_reward = float(gamma_reward)
        self.target_reward = float(target_reward)
        self.sensitivity = float(sensitivity)

    def init_cell_state(self, cell):
        """
        Initialize EMA state and starting alpha.
        """
        cell.reward_ema = 0.0
        mid = 0.5 * (self.alpha_min + self.alpha_max)
        cell.alpha_W = mid
        cell.alpha = mid

    def update_learning_rate(self, cell, reward_t: float):
        """
        Update alpha_W based on new scalar reward.

        Equations:
            R_t = γ R_{t-1} + (1-γ) r_t
            adv_t = R_t - target_reward
            λ_t = σ(-k * adv_t)
            α_t = α_min + λ_t (α_max - α_min)

        => reward > target → adv>0 → λ<0.5 → α toward alpha_min (conservative)
           reward < target → adv<0 → λ>0.5 → α toward alpha_max (aggressive)
        """
        r = float(reward_t)
        R_prev = float(getattr(cell, "reward_ema", 0.0))
        R_t = self.gamma_reward * R_prev + (1.0 - self.gamma_reward) * r
        cell.reward_ema = R_t

        adv = R_t - self.target_reward

        # lambda in (0,1)
        lam = _sigmoid(-self.sensitivity * adv)

        alpha_span = self.alpha_max - self.alpha_min
        alpha_t = self.alpha_min + lam * alpha_span
        alpha_t = _clip(alpha_t, self.alpha_min, self.alpha_max)

        cell.alpha_W = alpha_t
        cell.alpha = alpha_t

        return alpha_t


# ---------------------------------------------------------------------
# Genome Registry for Evolution Testing
# ---------------------------------------------------------------------

def create_evolution_genome_registry():
    """
    Create registry of genomes for regime-based evolution testing.
    
    Each genome type should dominate in specific regimes:
    - StaticGenome: Regime 0 (Stable & Clean)
    - PredictionGenome: Regime 1 (Slowly Drifting) 
    - RewardGenomeV2: Regime 2 (Abrupt Switches + Noise)
    - NeighborGenome: Regime 3 (Cooperative)
    """
    return {
        'static_conservative': StaticGenome(alpha=5e-4),
        'static_moderate': StaticGenome(alpha=1e-3),
        'static_aggressive': StaticGenome(alpha=2e-3),
        
        'prediction_conservative': PredictionGenome(
            alpha_min=1e-4, alpha_max=5e-3, sensitivity=2.0
        ),
        'prediction_moderate': PredictionGenome(
            alpha_min=5e-4, alpha_max=1e-2, sensitivity=3.0
        ),
        'prediction_aggressive': PredictionGenome(
            alpha_min=1e-3, alpha_max=2e-2, sensitivity=4.0
        ),
        
        'neighbor_conservative': NeighborGenome(
            alpha_min=1e-4, alpha_max=5e-3, neighbor_weight=0.3
        ),
        'neighbor_moderate': NeighborGenome(
            alpha_min=5e-4, alpha_max=1e-2, neighbor_weight=0.5
        ),
        'neighbor_aggressive': NeighborGenome(
            alpha_min=1e-3, alpha_max=2e-2, neighbor_weight=0.7
        ),
        
        'reward_v2_conservative': RewardGenomeV2(
            alpha_min=1e-4, alpha_max=5e-3, target_reward=0.6, sensitivity=3.0
        ),
        'reward_v2_moderate': RewardGenomeV2(
            alpha_min=5e-4, alpha_max=1e-2, target_reward=0.5, sensitivity=4.0
        ),
        'reward_v2_aggressive': RewardGenomeV2(
            alpha_min=1e-3, alpha_max=2e-2, target_reward=0.4, sensitivity=5.0
        ),
    }


if __name__ == "__main__":
    print("🧬 Evolution Genome Classes")
    print("=" * 30)
    
    genomes = create_evolution_genome_registry()
    
    print(f"Available genomes ({len(genomes)}):")
    for name, genome in genomes.items():
        genome_type = type(genome).__name__
        if hasattr(genome, 'alpha'):
            print(f"  {name}: {genome_type}(α={genome.alpha:.0e})")
        elif hasattr(genome, 'alpha_min'):
            print(f"  {name}: {genome_type}(α∈[{genome.alpha_min:.0e}, {genome.alpha_max:.0e}])")
    
    print("\nRegime expectations:")
    print("  Regime 0 (Stable): StaticGenome should dominate")
    print("  Regime 1 (Drifting): PredictionGenome should dominate") 
    print("  Regime 2 (Switches): RewardGenomeV2 should dominate")
    print("  Regime 3 (Cooperative): NeighborGenome should dominate")