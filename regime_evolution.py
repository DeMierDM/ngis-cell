"""
Regime-Based Evolution Environment 
================================

Implementation of 4-regime phased environment for testing genome evolution:

Regime 0: Stable & Clean (favors StaticGenome)
Regime 1: Slowly Drifting (favors PredictionGenome) 
Regime 2: Abrupt Switches + Noise (favors RewardGenomeV2)
Regime 3: Cooperative (favors NeighborGenome)

Each regime lasts T_regime steps, cycles repeat.
Evolution should show different genome dominance in different regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import numpy as np
import random
import math
from multivector import Multivector
from genomes import StaticGenome, PredictionGenome, NeighborGenome, RewardGenomeV2
from rotation_utils import apply_rotation, normalize_vector, random_unit_vector


@dataclass
class GenomeSpec:
    """Metadata + factory for a genome type."""
    name: str
    genome_ctor: Callable[..., Any]
    init_kwargs: Dict[str, Any]


# Genome registry for regime-based evolution
EVOLUTION_GENOME_REGISTRY = [
    # Static genomes (should dominate Regime 0)
    GenomeSpec(
        name="static_conservative",
        genome_ctor=lambda **kwargs: StaticGenome(**kwargs),
        init_kwargs={"alpha": 5e-4}
    ),
    GenomeSpec(
        name="static_moderate", 
        genome_ctor=lambda **kwargs: StaticGenome(**kwargs),
        init_kwargs={"alpha": 1e-3}
    ),
    
    # Prediction genomes (should dominate Regime 1)
    GenomeSpec(
        name="prediction_conservative",
        genome_ctor=lambda **kwargs: PredictionGenome(**kwargs),
        init_kwargs={"alpha_min": 1e-4, "alpha_max": 5e-3, "sensitivity": 2.0}
    ),
    GenomeSpec(
        name="prediction_moderate",
        genome_ctor=lambda **kwargs: PredictionGenome(**kwargs),
        init_kwargs={"alpha_min": 5e-4, "alpha_max": 1e-2, "sensitivity": 3.0}
    ),
    
    # Neighbor genomes (should dominate Regime 3)
    GenomeSpec(
        name="neighbor_conservative",
        genome_ctor=lambda **kwargs: NeighborGenome(**kwargs),
        init_kwargs={"alpha_min": 1e-4, "alpha_max": 5e-3, "neighbor_weight": 0.3}
    ),
    GenomeSpec(
        name="neighbor_moderate",
        genome_ctor=lambda **kwargs: NeighborGenome(**kwargs),
        init_kwargs={"alpha_min": 5e-4, "alpha_max": 1e-2, "neighbor_weight": 0.5}
    ),
    
    # Reward genomes (should dominate Regime 2)
    GenomeSpec(
        name="reward_v2_conservative",
        genome_ctor=lambda **kwargs: RewardGenomeV2(**kwargs),
        init_kwargs={"alpha_min": 1e-4, "alpha_max": 5e-3, "target_reward": 0.6, "sensitivity": 3.0}
    ),
    GenomeSpec(
        name="reward_v2_aggressive",
        genome_ctor=lambda **kwargs: RewardGenomeV2(**kwargs),
        init_kwargs={"alpha_min": 1e-3, "alpha_max": 2e-2, "target_reward": 0.4, "sensitivity": 5.0}
    ),
]


class EvolutionCell:
    """
    Simplified cell wrapper for regime-based evolution testing.
    
    Focuses on genome behavior rather than full GA dynamics.
    """
    
    def __init__(self, cell_id: int, genome_spec: GenomeSpec):
        self.id = cell_id
        self.genome_spec = genome_spec
        self.genome = genome_spec.genome_ctor(**genome_spec.init_kwargs)
        
        # Initialize genome state
        if hasattr(self.genome, 'init_cell_state'):
            self.genome.init_cell_state(self)
        
        # Learning state
        self.alpha_W = getattr(self, 'alpha_W', 1e-3)
        self.alpha = self.alpha_W
        
        # Internal prediction state (simple linear predictor)
        self.W = np.random.normal(0, 0.1, (3, 3)).astype(np.float32)
        
        # Fitness tracking
        self.total_reward = 0.0
        self.steps_alive = 0
        self.recent_rewards: List[float] = []
        self.last_prediction = np.zeros(3)
        self.last_reward = 0.0

    @property
    def genome_name(self):
        return self.genome_spec.name

    def predict(self, x_t: np.ndarray) -> np.ndarray:
        """Simple linear prediction."""
        x_norm = x_t / (np.linalg.norm(x_t) + 1e-8)
        pred = self.W @ x_norm
        pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
        self.last_prediction = pred_norm
        return pred_norm

    def learn(self, x_t: np.ndarray, y_true: np.ndarray, reward_external: Optional[float] = None):
        """
        Learn from prediction task with genome-specific adaptation.
        """
        # 1) Predict
        y_hat = self.predict(x_t)
        
        # 2) Compute error and reward
        err_vec = y_true - y_hat
        err = float(np.linalg.norm(err_vec))
        
        if reward_external is None:
            beta = 10.0
            reward = math.exp(-beta * err)
        else:
            reward = float(reward_external)
        
        # 3) Update learning rate based on genome type
        if isinstance(self.genome, PredictionGenome):
            self.genome.update_learning_rate(self, prediction_error=err)
        elif isinstance(self.genome, RewardGenomeV2):
            self.genome.update_learning_rate(self, reward_t=reward)
        elif isinstance(self.genome, StaticGenome):
            self.genome.update_learning_rate(self)  # No-op
        # NeighborGenome uses update_from_network_signals in environment
        
        # 4) Apply gradient update
        alpha = float(getattr(self, "alpha_W", 1e-3))
        
        # Simple gradient: dL/dW = -2 * err_vec * x_t^T
        x_norm = x_t / (np.linalg.norm(x_t) + 1e-8)
        grad_W = -2.0 * np.outer(err_vec, x_norm)
        
        self.W = self.W - alpha * grad_W
        
        # 5) Track statistics
        self.last_reward = reward
        self.total_reward += reward
        self.steps_alive += 1
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        return err**2, reward

    def average_reward(self, horizon: int = 100) -> float:
        if not self.recent_rewards:
            return 0.0
        return float(np.mean(self.recent_rewards[-horizon:]))

    def clone_with_mutation(self, new_id: int, mutation_prob: float = 0.1):
        """Create mutated clone for selection."""
        new_kwargs = dict(self.genome_spec.init_kwargs)
        
        if random.random() < mutation_prob:
            for k, v in new_kwargs.items():
                if isinstance(v, (int, float)):
                    noise = np.random.normal(loc=0.0, scale=0.1 * abs(v) if v != 0 else 1e-4)
                    new_kwargs[k] = v + noise
        
        new_spec = GenomeSpec(
            name=self.genome_spec.name,
            genome_ctor=self.genome_spec.genome_ctor,
            init_kwargs=new_kwargs,
        )
        
        new_cell = EvolutionCell(new_id, new_spec)
        return new_cell


class RegimeBasedEvolutionEnv:
    """
    4-regime evolution environment exactly as specified.
    
    T_regime = 2000 steps per regime
    Regime cycle: 0 -> 1 -> 2 -> 3 -> 0 -> ...
    """
    
    def __init__(
        self,
        num_cells: int = 40,
        T_regime: int = 2000,
        K_switch: int = 200,  # For Regime 2 abrupt switches
        selection_interval: int = 400,
        replacement_fraction: float = 0.25,
        neighbor_topology: str = "ring",
    ):
        self.num_cells = num_cells
        self.T_regime = T_regime
        self.K_switch = K_switch
        self.selection_interval = selection_interval
        self.replacement_fraction = replacement_fraction
        
        # Initialize population
        self.cells: List[EvolutionCell] = []
        self.global_step = 0
        
        # Environment state
        self.v_t = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Current vector
        
        # Regime 2 state (abrupt switches)
        self._axis_reg2 = None
        self._omega_reg2 = None
        
        # Initialize population and topology
        self._init_population()
        self.neighbors = self._build_topology(neighbor_topology)
        
        # Logging
        self.history = {
            "step": [],
            "regime_id": [],
            "genome_counts": [],
            "genome_avg_reward": [],
            "global_avg_reward": [],
        }

    def _init_population(self):
        """Initialize with random genome distribution."""
        for i in range(self.num_cells):
            spec = random.choice(EVOLUTION_GENOME_REGISTRY)
            cell = EvolutionCell(i, spec)
            self.cells.append(cell)

    def _build_topology(self, topology: str):
        """Build neighbor topology."""
        neighbors = {}
        if topology == "ring":
            for i in range(self.num_cells):
                left = (i - 1) % self.num_cells
                right = (i + 1) % self.num_cells
                neighbors[i] = [left, right]
        elif topology == "fully_connected":
            for i in range(self.num_cells):
                neighbors[i] = [j for j in range(self.num_cells) if j != i]
        return neighbors

    def _sample_task(self, step: int):
        """
        Sample task according to 4-regime specification.
        
        Returns (x_t, task_info) where task_info contains target_vector
        """
        regime_id = (step // self.T_regime) % 4
        
        if regime_id == 0:
            # Regime 0: Stable & Clean
            omega = 0.01
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            noise_sigma = 0.01
            
        elif regime_id == 1:
            # Regime 1: Slowly Drifting
            theta = 0.001 * step
            axis = np.array([np.cos(theta), np.sin(theta), 0.3], dtype=np.float32)
            axis = normalize_vector(axis)
            omega = 0.02 + 0.01 * np.sin(0.0005 * step)
            noise_sigma = 0.05
            
        elif regime_id == 2:
            # Regime 2: Abrupt Switches + High Noise
            if (step % self.K_switch) == 0 or self._axis_reg2 is None:
                self._axis_reg2 = random_unit_vector()
                self._omega_reg2 = np.random.uniform(0.01, 0.1)
            axis = self._axis_reg2
            omega = self._omega_reg2
            noise_sigma = 0.15
            
        elif regime_id == 3:
            # Regime 3: Cooperative
            axis = np.array([0.0, 1.0, 0.3], dtype=np.float32)
            axis = normalize_vector(axis)
            omega = 0.015
            noise_sigma = 0.05
        
        # Apply rotation dynamics
        v_next = apply_rotation(self.v_t, axis, omega)
        v_next = v_next + np.random.normal(scale=noise_sigma, size=3).astype(np.float32)
        
        x_t = self.v_t.copy()
        self.v_t = v_next.copy()
        
        task_info = {
            "regime_id": regime_id,
            "axis": axis,
            "omega": omega,
            "noise_sigma": noise_sigma,
            "target_vector": v_next,
        }
        
        return x_t, task_info

    def _compute_cooperative_rewards(self, base_rewards: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Compute cooperative rewards for Regime 3.
        
        r_i = base_reward_i * (1 - λ_social) + λ_social * coordination_bonus_i
        """
        lambda_social = 0.4
        gamma = 5.0  # coordination sharpness
        
        cooperative_rewards = base_rewards.copy()
        
        for i in range(self.num_cells):
            nbr_indices = self.neighbors[i]
            if not nbr_indices:
                continue
            
            # Compute coordination with neighbors
            my_pred = predictions[i]
            neighbor_preds = predictions[nbr_indices]
            
            # Average alignment with neighbors
            alignment_scores = []
            for nbr_pred in neighbor_preds:
                # Cosine similarity (higher = better alignment)
                cos_sim = np.dot(my_pred, nbr_pred) / (
                    np.linalg.norm(my_pred) * np.linalg.norm(nbr_pred) + 1e-8
                )
                alignment_scores.append(max(0, cos_sim))  # Only positive alignment
            
            avg_alignment = np.mean(alignment_scores) if alignment_scores else 0
            coordination_bonus = math.exp(gamma * (avg_alignment - 0.5))  # Bonus for alignment > 0.5
            
            # Combine base reward with cooperation bonus
            base_reward = base_rewards[i]
            cooperative_rewards[i] = (
                base_reward * (1 - lambda_social) + 
                lambda_social * coordination_bonus * base_reward
            )
        
        return cooperative_rewards

    def step(self):
        """Single environment step with regime-specific dynamics."""
        self.global_step += 1
        x_t, task_info = self._sample_task(self.global_step)
        regime_id = task_info["regime_id"]
        
        # 1) Each cell learns
        losses = []
        base_rewards = []
        predictions = []
        
        for i, cell in enumerate(self.cells):
            loss, reward = cell.learn(x_t, task_info["target_vector"])
            losses.append(loss)
            base_rewards.append(reward)
            predictions.append(cell.last_prediction.copy())
        
        losses = np.array(losses)
        base_rewards = np.array(base_rewards)
        predictions = np.array(predictions)
        
        # 2) Regime-specific reward adjustment
        if regime_id == 3:  # Cooperative regime
            final_rewards = self._compute_cooperative_rewards(base_rewards, predictions)
            # Update cell rewards for cooperative regime
            for i, cell in enumerate(self.cells):
                cell.last_reward = final_rewards[i]
                cell.total_reward += (final_rewards[i] - base_rewards[i])  # Add bonus
        else:
            final_rewards = base_rewards
        
        # 3) Network signals for NeighborGenome
        global_avg_reward = float(np.mean(final_rewards))
        neighbor_avg_rewards = []
        
        for i in range(self.num_cells):
            nbr_indices = self.neighbors[i]
            if nbr_indices:
                neighbor_avg_rewards.append(float(np.mean(final_rewards[nbr_indices])))
            else:
                neighbor_avg_rewards.append(global_avg_reward)
        
        # Update NeighborGenomes
        for i, cell in enumerate(self.cells):
            if isinstance(cell.genome, NeighborGenome):
                cell.genome.update_from_network_signals(
                    cell, global_avg_reward, neighbor_avg_rewards[i], self.global_step
                )
        
        # 4) Logging
        self._log_step(regime_id, global_avg_reward)
        
        # 5) Selection
        if self.global_step % self.selection_interval == 0:
            self._selection_round()

    def _log_step(self, regime_id: int, global_avg_reward: float):
        """Log evolution statistics."""
        self.history["step"].append(self.global_step)
        self.history["regime_id"].append(regime_id)
        self.history["global_avg_reward"].append(global_avg_reward)
        
        # Genome counts and performance
        names = [cell.genome_name for cell in self.cells]
        unique_names = sorted(set(names))
        counts = {name: names.count(name) for name in unique_names}
        
        genome_rewards = {name: [] for name in unique_names}
        for cell in self.cells:
            genome_rewards[cell.genome_name].append(cell.average_reward())
        
        genome_avg_reward = {
            name: float(np.mean(rewards)) if rewards else 0.0
            for name, rewards in genome_rewards.items()
        }
        
        self.history["genome_counts"].append(counts)
        self.history["genome_avg_reward"].append(genome_avg_reward)

    def _selection_round(self):
        """Selection with mutation."""
        horizon = min(200, self.selection_interval)
        cells_with_fitness = [
            (i, cell.average_reward(horizon))
            for i, cell in enumerate(self.cells)
        ]
        
        cells_with_fitness.sort(key=lambda t: t[1], reverse=True)
        
        num_replace = int(self.replacement_fraction * self.num_cells)
        if num_replace < 1:
            return
        
        winners = cells_with_fitness[:-num_replace]
        losers = cells_with_fitness[-num_replace:]
        
        next_id = max(cell.id for cell in self.cells) + 1
        
        for (loser_idx, _) in losers:
            winner_idx, _ = random.choice(winners)
            winner_cell = self.cells[winner_idx]
            
            new_cell = winner_cell.clone_with_mutation(next_id, mutation_prob=0.15)
            next_id += 1
            
            self.cells[loser_idx] = new_cell

    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable regime name."""
        regime_names = {
            0: "Stable & Clean",
            1: "Slowly Drifting", 
            2: "Abrupt Switches",
            3: "Cooperative"
        }
        return regime_names.get(regime_id, "Unknown")


def run_evolution_experiment(num_cycles: int = 3, num_cells: int = 30):
    """
    Run complete evolution experiment across regime cycles.
    """
    print("🧬 REGIME-BASED EVOLUTION EXPERIMENT")
    print("=" * 50)
    
    env = RegimeBasedEvolutionEnv(num_cells=num_cells)
    total_steps = num_cycles * 4 * env.T_regime
    
    print(f"Population: {num_cells} cells")
    print(f"Regime length: {env.T_regime} steps")
    print(f"Total cycles: {num_cycles}")
    print(f"Total steps: {total_steps}")
    
    # Show initial distribution
    initial_counts = {}
    for cell in env.cells:
        name = cell.genome_name
        initial_counts[name] = initial_counts.get(name, 0) + 1
    
    print(f"\nInitial distribution:")
    for name, count in sorted(initial_counts.items()):
        print(f"  {name}: {count}")
    
    print(f"\n🏃 Running evolution...")
    
    regime_progress = {}
    current_regime = -1
    
    for step in range(total_steps):
        env.step()
        
        regime_id = (step // env.T_regime) % 4
        
        # Track regime transitions
        if regime_id != current_regime:
            current_regime = regime_id
            regime_name = env.get_regime_name(regime_id)
            print(f"\nStep {step}: Entering {regime_name} (Regime {regime_id})")
            
            # Show current distribution
            current_counts = {}
            for cell in env.cells:
                name = cell.genome_name
                current_counts[name] = current_counts.get(name, 0) + 1
            
            for name, count in sorted(current_counts.items()):
                pct = 100 * count / num_cells
                print(f"  {name}: {count} ({pct:.1f}%)")
        
        # Progress within regime
        regime_step = step % env.T_regime
        if regime_step in [500, 1000, 1500]:
            progress = 100 * regime_step / env.T_regime
            avg_reward = np.mean([cell.average_reward() for cell in env.cells])
            print(f"  {progress:.0f}% through regime - avg reward: {avg_reward:.4f}")
    
    # Final results
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 40)
    
    final_counts = {}
    final_rewards = {}
    for cell in env.cells:
        name = cell.genome_name
        final_counts[name] = final_counts.get(name, 0) + 1
        if name not in final_rewards:
            final_rewards[name] = []
        final_rewards[name].append(cell.average_reward())
    
    sorted_results = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, count) in enumerate(sorted_results, 1):
        avg_reward = np.mean(final_rewards[name])
        percentage = 100 * count / num_cells
        print(f"  {i}. {name}: {count}/{num_cells} ({percentage:.1f}%) - reward: {avg_reward:.4f}")
    
    return env


if __name__ == "__main__":
    # Test rotation utilities first
    print("🔄 Testing rotation utilities...")
    from rotation_utils import apply_rotation
    v = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    v_rot = apply_rotation(v, axis, math.pi/2)
    print(f"Rotation test: [1,0,0] -> {v_rot} (expected ~[0,1,0])")
    
    # Run evolution experiment
    env = run_evolution_experiment(num_cycles=2, num_cells=24)
    print("\n✅ Evolution experiment complete!")