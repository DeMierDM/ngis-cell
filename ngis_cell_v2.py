#!/usr/bin/env python3
"""
NGISCell v2 - Fixed Reward Genome Architecture
==============================================

Implements the proper bounded EMA reward modulation that prevents α→0 collapse
while maintaining strong genome-driven behavioral differences.

Key improvements:
- RewardGenomeV2 with discounted reward (EMA)
- Bounded learning rates [α_min, α_max]  
- Smooth sigmoid modulation instead of exponential death spiral
- Proper advantage-based learning (reward vs target, not raw accumulation)

Mathematical foundation:
    R_t = γ R_{t-1} + (1-γ) r_t                    # EMA reward smoothing
    adv_t = R_t - R_target                         # advantage signal  
    λ_t = σ(-k * adv_t) = 1/(1 + exp(k * adv_t))  # sigmoid mapping
    α_t = α_min + λ_t (α_max - α_min)             # bounded learning rate

This ensures α_t ∈ [α_min, α_max] always, preventing both collapse and explosion.
"""

from dataclasses import dataclass, field
import numpy as np
import math
from typing import Optional, Dict, List, Any
from multivector import Multivector, rotor_from_vectors, apply_rotor


# =====================================================
# 1. GENOME DEFINITIONS (v2 with proper reward handling)
# =====================================================

@dataclass
class NGISGenome:
    """Base genome class - controls HOW this cell learns."""
    rule_type: str = "pure_pred"  # "pure_pred", "neighbor_aware", "reward_v1", "reward_v2"
    use_pred_error: bool = True
    use_neighbor_error: bool = False
    use_reward: bool = False
    
    # Base learning rate bounds
    alpha_W_min: float = 1e-4
    alpha_W_max: float = 5e-3
    
    # Neighbor mixing
    neighbor_mix_weight: float = 0.3
    
    # Rotor learning parameters
    rotor_learning_rate: float = 0.1


@dataclass 
class RewardGenomeV1(NGISGenome):
    """Original collapsing reward genome - kept as 'bad species' example."""
    rule_type: str = "reward_v1"
    use_reward: bool = True
    reward_sensitivity: float = 2.0  # This causes the α→0 collapse


class RewardGenomeV2(NGISGenome):
    """
    Fixed reward genome with bounded EMA modulation.
    
    Mathematical behavior:
    - R_t smoothed via EMA (no infinite accumulation)
    - α_t bounded in [α_min, α_max] (no collapse/explosion)  
    - Smooth sigmoid response (no harsh exponentials)
    - Advantage-based (reward relative to target, not absolute)
    - Philosophy A: Success stabilizes learning, failure increases exploration
    """
    
    def __init__(
        self,
        alpha_min=1e-4,
        alpha_max=1e-1, 
        base_alpha=None,
        gamma_reward=0.95,      # EMA smoothing factor
        target_reward=0.5,      # what constitutes 'good' reward
        sensitivity=3.0,        # how sharply reward affects α (reduced from 5.0)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rule_type = "reward_v2"
        self.use_reward = True
        
        assert 0.0 < alpha_min <= alpha_max
        assert 0.0 < gamma_reward < 1.0
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max 
        self.base_alpha = base_alpha
        self.gamma_reward = gamma_reward
        self.target_reward = target_reward
        self.sensitivity = sensitivity

    @staticmethod
    def _sigmoid(x):
        """Backend-agnostic sigmoid."""
        if hasattr(x, "exp"):  # torch tensor
            return 1.0 / (1.0 + (-x).exp())
        else:
            return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def _clip(x, x_min, x_max):
        """Backend-agnostic clipping.""" 
        if hasattr(x, "clamp"):  # torch tensor
            return x.clamp(x_min, x_max)
        else:
            return max(x_min, min(x, x_max))
    
    def init_cell_state(self, cell):
        """Initialize cell state for reward modulation."""
        cell.reward_ema = 0.0
        
        if self.base_alpha is not None:
            cell.alpha_W = self._clip(self.base_alpha, self.alpha_min, self.alpha_max)
        else:
            mid = 0.5 * (self.alpha_min + self.alpha_max)
            cell.alpha_W = mid
    
    def update_learning_rate(self, cell, reward_t):
        """
        Update cell reward state and learning rate based on bounded EMA.
        
        Returns:
            tuple: (alpha_t, R_t) for logging
        """
        # 1) Update discounted reward (EMA)
        R_prev = float(getattr(cell, "reward_ema", 0.0))
        R_t = self.gamma_reward * R_prev + (1.0 - self.gamma_reward) * float(reward_t)
        cell.reward_ema = R_t
        
        # 2) Compute advantage (how much better/worse than target)
        adv_t = R_t - self.target_reward
        
        # 3) Map advantage to [0,1] via sigmoid
        # Philosophy A: "Success = stabilize, failure = adapt harder"
        # Good performance (adv > 0) → λ < 0.5 → α closer to α_min (stabilize)
        # Bad performance (adv < 0) → λ > 0.5 → α closer to α_max (adapt harder)
        lam = self._sigmoid(-self.sensitivity * adv_t)
        
        # 4) Map λ to bounded learning rate
        alpha_span = self.alpha_max - self.alpha_min
        alpha_t = self.alpha_min + lam * alpha_span
        
        # 5) Set on cell
        cell.alpha_W = alpha_t
        
        return alpha_t, R_t


# =====================================================
# 2. CELL STATE
# =====================================================

@dataclass
class NGISCellState:
    """Fast-changing state (cytoplasm)."""
    psi: np.ndarray                    # Cl(3,0) multivector coeffs [8]
    W_rotor: Multivector              # Local rotor operator
    alpha_W: float                    # Current learning rate
    step: int = 0
    
    # History buffers for adaptive learning rules
    pred_error_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list) 
    neighbor_error_history: List[float] = field(default_factory=list)
    
    # v2 reward genome state
    reward_ema: float = 0.0


# =====================================================
# 3. NGIS CELL v2
# =====================================================

class NGISCell:
    """
    Nucleus + Genome controlled cell with fixed reward modulation.
    
    Key fix: RewardGenomeV2 prevents α→0 collapse via bounded EMA.
    """
    
    def __init__(self, cell_id: int, genome: NGISGenome, initial_state: NGISCellState):
        self.cell_id = cell_id
        self.G = genome  # nucleus
        self.s = initial_state  # cytoplasm
        
        # Initialize genome-specific state
        if hasattr(self.G, 'init_cell_state'):
            self.G.init_cell_state(self)
    
    @classmethod
    def create_random(cls, cell_id: int, genome: NGISGenome) -> 'NGISCell':
        """Create cell with random initialization."""
        # Random multivector state
        psi = np.random.randn(8) * 0.1
        psi[0] = 1.0  # scalar part
        
        # Random rotor (small rotation)
        angle = np.random.uniform(0, np.pi/6)  # up to 30 degrees
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        cos_half = np.cos(angle/2)
        sin_half = np.sin(angle/2)
        bivector = sin_half * axis
        
        rotor_coeffs = np.zeros(8)
        rotor_coeffs[0] = cos_half
        rotor_coeffs[4:7] = bivector  # e23, e13, e12 components
        W_rotor = Multivector(rotor_coeffs)
        
        initial_alpha = genome.alpha_W_min + 0.5 * (genome.alpha_W_max - genome.alpha_W_min)
        
        state = NGISCellState(
            psi=psi,
            W_rotor=W_rotor, 
            alpha_W=initial_alpha
        )
        
        return cls(cell_id, genome, state)
    
    def _normalize_rotor(self):
        """Ensure W_rotor remains a valid unit rotor."""
        scalar_part = self.s.W_rotor.c[0]
        bivector_parts = self.s.W_rotor.c[4:7] 
        
        norm_sq = scalar_part**2 + np.sum(bivector_parts**2)
        norm = np.sqrt(norm_sq + 1e-8)
        
        self.s.W_rotor = Multivector(self.s.W_rotor.c / norm)
    
    def sense(self, x_t: np.ndarray, neighbor_msgs: List[Dict[str, Any]]):
        """Update psi from input + neighbor information."""
        # Normalize input
        x_norm = x_t / (np.linalg.norm(x_t) + 1e-8)
        
        # Convert to GA vector
        x_mv = Multivector.from_vector3(x_norm)
        
        # Blend into current psi state
        current_psi_mv = Multivector(self.s.psi)
        
        # Simple blending: 80% current, 20% new input
        blend_factor = 0.2
        new_psi = current_psi_mv * (1 - blend_factor) + x_mv * blend_factor
        
        # Optional: mix with neighbors if available
        if neighbor_msgs and len(neighbor_msgs) > 0:
            neighbor_psis = [msg["psi"] for msg in neighbor_msgs if "psi" in msg]
            if neighbor_psis:
                neighbor_avg = np.mean(neighbor_psis, axis=0)
                neighbor_mv = Multivector(neighbor_avg)
                
                # Light neighbor influence
                neighbor_weight = 0.1
                new_psi = new_psi * (1 - neighbor_weight) + neighbor_mv * neighbor_weight
        
        self.s.psi = new_psi.c
    
    def predict(self) -> np.ndarray:
        """Generate prediction using current rotor and psi state."""
        # Extract vector part from psi
        current_vec = self.s.psi[1:4]
        
        # Apply rotor transformation: R * vec * R~
        vec_mv = Multivector.from_vector3(current_vec)
        rotated_mv = apply_rotor(self.s.W_rotor, vec_mv)
        
        return rotated_mv.c[1:4]  # Return vector part
    
    def learn(self, target: np.ndarray, reward: float, neighbor_stats: Optional[Dict[str, Any]]):
        """Genome-controlled learning with v2 reward handling."""
        # 1) Compute prediction error
        pred = self.predict()
        error_vec = pred - target
        error_magnitude = float(np.linalg.norm(error_vec))
        
        # Log histories
        self.s.pred_error_history.append(error_magnitude)
        self.s.reward_history.append(reward)
        
        if neighbor_stats and "mean_error" in neighbor_stats:
            self.s.neighbor_error_history.append(neighbor_stats["mean_error"])
        
        # 2) Update learning rate based on genome type
        if hasattr(self.G, 'update_learning_rate'):
            # v2 reward genome with bounded EMA
            alpha_t, R_t = self.G.update_learning_rate(self, reward)
        elif self.G.rule_type == "reward_v1":
            # v1 collapsing genome (for comparison)
            self._update_reward_v1(reward)
        
        # 3) Select update rule based on genome
        if self.G.rule_type in ["pure_pred", "reward_v1", "reward_v2"]:
            self._update_W_pure_pred(error_vec, target)
        elif self.G.rule_type == "neighbor_aware":
            self._update_W_neighbor_aware(error_vec, target, neighbor_stats)
        
        self.s.step += 1
    
    def _update_reward_v1(self, reward: float):
        """Original collapsing reward modulation (for comparison)."""
        if not hasattr(self.G, 'reward_sensitivity'):
            return
        
        baseline = np.mean(self.s.reward_history[-10:]) if len(self.s.reward_history) >= 10 else 0.0
        delta_r = reward - baseline
        
        # This causes the collapse: α *= exp(-k * δ)  
        self.s.alpha_W *= np.exp(-self.G.reward_sensitivity * delta_r)
        self.s.alpha_W = max(self.s.alpha_W, self.G.alpha_W_min)  # weak safety
    
    def _update_W_pure_pred(self, error_vec: np.ndarray, target: np.ndarray):
        """Pure prediction-based rotor update."""
        if not self.G.use_pred_error:
            return
        
        # Current prediction direction
        pred = self.predict()
        pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        
        # Compute correction rotor to align pred → target
        try:
            correction_rotor, angle = rotor_from_vectors(pred_norm, target_norm)
            
            # Apply small correction: R_new = correction^α * R_old
            identity = Multivector.scalar(1.0)
            alpha = self.s.alpha_W * self.G.rotor_learning_rate
            
            # Linear interpolation in rotor space (not geometrically perfect but stable)
            blended_correction = identity * (1 - alpha) + correction_rotor * alpha
            
            # Compose: W = blended_correction * W
            self.s.W_rotor = blended_correction * self.s.W_rotor
            self._normalize_rotor()
            
        except:
            # Fallback: small random perturbation
            small_angle = self.s.alpha_W * 0.1
            random_axis = np.random.randn(3)
            random_axis = random_axis / (np.linalg.norm(random_axis) + 1e-8)
            
            cos_half = np.cos(small_angle/2)
            sin_half = np.sin(small_angle/2)
            
            small_rotor_coeffs = np.zeros(8)
            small_rotor_coeffs[0] = cos_half
            small_rotor_coeffs[4:7] = sin_half * random_axis
            
            small_rotor = Multivector(small_rotor_coeffs)
            self.s.W_rotor = small_rotor * self.s.W_rotor
            self._normalize_rotor()
    
    def _update_W_neighbor_aware(self, error_vec: np.ndarray, target: np.ndarray, neighbor_stats: Optional[Dict[str, Any]]):
        """Neighbor-aware rotor update."""
        # Start with pure prediction update
        self._update_W_pure_pred(error_vec, target)
        
        # Modulate learning rate based on neighbor comparison
        if (self.G.use_neighbor_error and neighbor_stats and 
            "mean_error" in neighbor_stats and len(self.s.pred_error_history) > 0):
            
            local_error = self.s.pred_error_history[-1]
            neighbor_error = neighbor_stats["mean_error"]
            
            # If we're doing worse than neighbors, learn faster
            if neighbor_error > 0:
                error_ratio = (local_error + 1e-8) / (neighbor_error + 1e-8)
                self.s.alpha_W *= (1.0 + self.G.neighbor_mix_weight * (error_ratio - 1.0))
                self.s.alpha_W = np.clip(self.s.alpha_W, self.G.alpha_W_min, self.G.alpha_W_max)


# =====================================================
# 4. UTILITY FUNCTIONS FOR EXPERIMENTS
# =====================================================

def create_genome_library():
    """Create library of test genomes including v1 vs v2 reward variants."""
    genomes = {}
    
    # Baseline: pure prediction
    genomes['conservative'] = NGISGenome(
        rule_type="pure_pred",
        alpha_W_min=1e-5,
        alpha_W_max=1e-3,
        rotor_learning_rate=0.05
    )
    
    # Aggressive learner
    genomes['aggressive'] = NGISGenome(
        rule_type="pure_pred", 
        alpha_W_min=1e-3,
        alpha_W_max=1e-1,
        rotor_learning_rate=0.2
    )
    
    # Neighbor-aware
    genomes['neighbor_aware'] = NGISGenome(
        rule_type="neighbor_aware",
        use_neighbor_error=True,
        neighbor_mix_weight=0.5,
        alpha_W_min=1e-4,
        alpha_W_max=5e-3
    )
    
    # v1 collapsing reward (for comparison)
    genomes['reward_v1_collapse'] = RewardGenomeV1(
        alpha_W_min=1e-5,
        alpha_W_max=1e-1,
        reward_sensitivity=2.0  # causes collapse
    )
    
    # v2 bounded reward genomes with different characteristics
    genomes['reward_v2_conservative'] = RewardGenomeV2(
        alpha_min=1e-4,
        alpha_max=1e-2,
        gamma_reward=0.98,    # very smooth
        target_reward=0.5,
        sensitivity=2.0       # moderate sensitivity
    )
    
    genomes['reward_v2_aggressive'] = RewardGenomeV2(
        alpha_min=1e-3, 
        alpha_max=5e-2,
        gamma_reward=0.90,    # more responsive  
        target_reward=0.7,    # higher expectations
        sensitivity=4.0       # high sensitivity
    )
    
    genomes['reward_v2_adaptive'] = RewardGenomeV2(
        alpha_min=5e-4,
        alpha_max=2e-2, 
        gamma_reward=0.95,
        target_reward=0.3,    # lower expectations, easier to satisfy
        sensitivity=3.0
    )
    
    return genomes


if __name__ == "__main__":
    print("NGISCell v2 - Fixed Reward Genome Architecture")
    print("=" * 50)
    
    # Test genome creation
    genomes = create_genome_library()
    
    print(f"\nAvailable genomes ({len(genomes)}):")
    for name, genome in genomes.items():
        print(f"  {name}: {genome.rule_type}")
        if hasattr(genome, 'alpha_min'):
            print(f"    α ∈ [{genome.alpha_min:.0e}, {genome.alpha_max:.0e}]")
    
    # Test cell creation with v2 genome
    print(f"\nTesting RewardGenomeV2 cell creation...")
    test_genome = genomes['reward_v2_adaptive']
    test_cell = NGISCell.create_random(0, test_genome)
    
    print(f"  Initial α: {test_cell.s.alpha_W:.4f}")
    print(f"  Initial reward_ema: {test_cell.s.reward_ema:.3f}")
    
    # Test reward modulation
    print(f"\nTesting reward modulation (should stay bounded):")
    rewards = [0.8, 0.9, 0.6, 0.4, 0.2, 0.7, 0.8, 0.9]  # mix of good/bad
    
    for i, reward in enumerate(rewards):
        alpha_t, R_t = test_genome.update_learning_rate(test_cell, reward)
        print(f"  Step {i+1}: reward={reward:.1f} → α={alpha_t:.4f}, R_ema={R_t:.3f}")
    
    print(f"\n✅ v2 genome maintains α ∈ [{test_genome.alpha_min:.0e}, {test_genome.alpha_max:.0e}]")
    print("Ready for comparative experiments!")