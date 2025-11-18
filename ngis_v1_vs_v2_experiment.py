#!/usr/bin/env python3
"""
NGISCell v1 vs v2 Genome Comparison Experiment  
==============================================

This experiment directly compares the original collapsing reward genome (v1)
with the new bounded EMA reward genome (v2) to prove the fix works.

What we're testing:
1. Does v1 still collapse to α→0? (should: YES)
2. Does v2 stay bounded? (should: YES)  
3. Do v2 genomes still show meaningful behavioral differences? (should: YES)
4. Which v2 variant adapts best to regime changes?

Expected results:
- v1: dramatic performance then α collapse → learning death
- v2_conservative: steady, bounded behavior
- v2_aggressive: higher variance but bounded  
- v2_adaptive: best adaptation to changing rewards

This is the definitive test of whether the bounded EMA fix actually works.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple
from ngis_cell_v2 import NGISCell, create_genome_library


class RotatingVectorEnvironment:
    """
    Environment with regime changes to test genome adaptation.
    
    Regimes:
    - Phase 1 (0-300): Slow rotation, low noise
    - Phase 2 (300-600): Fast rotation, medium noise  
    - Phase 3 (600-900): Slow rotation, high noise
    - Phase 4 (900-1200): Medium rotation, low noise
    """
    
    def __init__(self, total_steps=1200):
        self.total_steps = total_steps
        self.step = 0
        self.current_angle = 0.0
        self.current_vector = np.array([1.0, 0.0, 0.0])
        
    def get_regime_params(self, step: int) -> Tuple[float, float]:
        """Return (rotation_speed, noise_level) for current regime."""
        if step < 300:
            return 0.02, 0.05    # slow rotation, low noise
        elif step < 600:  
            return 0.08, 0.15    # fast rotation, medium noise
        elif step < 900:
            return 0.02, 0.25    # slow rotation, high noise  
        else:
            return 0.05, 0.05    # medium rotation, low noise
    
    def step_environment(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Advance environment one step.
        
        Returns:
            current_vector: what cells observe
            next_vector: prediction target
            reward: performance-based reward signal
        """
        rotation_speed, noise_level = self.get_regime_params(self.step)
        
        # Current state (with noise)
        noise = np.random.randn(3) * noise_level
        current_noisy = self.current_vector + noise
        current_noisy = current_noisy / (np.linalg.norm(current_noisy) + 1e-8)
        
        # True next state
        self.current_angle += rotation_speed
        next_true = np.array([
            np.cos(self.current_angle),
            np.sin(self.current_angle),
            0.0
        ])
        
        # Update internal state
        self.current_vector = next_true
        
        # Reward based on regime (simulating task difficulty/performance)
        if self.step < 300:
            base_reward = 0.7  # easy regime
        elif self.step < 600:
            base_reward = 0.4  # hard regime (fast changes)
        elif self.step < 900:  
            base_reward = 0.3  # very hard (high noise)
        else:
            base_reward = 0.6  # recovery regime
        
        # Add some randomness to reward
        reward = base_reward + np.random.randn() * 0.1
        reward = np.clip(reward, 0.0, 1.0)
        
        self.step += 1
        return current_noisy, next_true, reward
    
    def reset(self):
        """Reset environment to initial state."""
        self.step = 0
        self.current_angle = 0.0
        self.current_vector = np.array([1.0, 0.0, 0.0])


def run_v1_vs_v2_experiment():
    """
    Run the definitive comparison: v1 (collapse) vs v2 (bounded) genomes.
    """
    print("🧪 NGIS Cell v1 vs v2 Genome Comparison")
    print("=" * 50)
    
    # Get genome library
    genomes = create_genome_library()
    
    # Select test genomes: v1 collapsing + 3 v2 variants
    test_genomes = {
        'v1_collapsing': genomes['reward_v1_collapse'],
        'v2_conservative': genomes['reward_v2_conservative'],  
        'v2_aggressive': genomes['reward_v2_aggressive'],
        'v2_adaptive': genomes['reward_v2_adaptive']
    }
    
    print(f"Testing {len(test_genomes)} genome variants:")
    for name, genome in test_genomes.items():
        print(f"  {name}: {genome.rule_type}")
        if hasattr(genome, 'alpha_min'):
            print(f"    α bounds: [{genome.alpha_min:.0e}, {genome.alpha_max:.0e}]")
    
    # Initialize environment and cells
    env = RotatingVectorEnvironment(total_steps=1200)
    cells = {}
    
    for name, genome in test_genomes.items():
        cells[name] = NGISCell.create_random(0, genome)
    
    # Data collection
    results = {name: {
        'errors': [],
        'learning_rates': [],
        'reward_emas': [],
        'predictions': [],
        'rewards': []
    } for name in test_genomes.keys()}
    
    results['environment'] = {
        'regime_changes': [0, 300, 600, 900, 1200],
        'inputs': [],
        'targets': [],
        'rewards': []
    }
    
    print(f"\n🚀 Running experiment for {env.total_steps} steps...")
    start_time = time.time()
    
    # Main experiment loop
    for step in range(env.total_steps):
        # Environment step
        current_input, target, reward = env.step_environment()
        
        results['environment']['inputs'].append(current_input.tolist())
        results['environment']['targets'].append(target.tolist()) 
        results['environment']['rewards'].append(reward)
        
        # Update each cell
        for name, cell in cells.items():
            # Sense current input
            cell.sense(current_input, [])
            
            # Make prediction
            prediction = cell.predict()
            results[name]['predictions'].append(prediction.tolist())
            
            # Compute error
            error = np.linalg.norm(prediction - target)
            results[name]['errors'].append(error)
            
            # Learn (with reward for reward-sensitive genomes)
            if cell.G.use_reward:
                cell.learn(target, reward, None)
            else:
                cell.learn(target, 0.0, None)  # non-reward genomes ignore reward
            
            # Log learning rate and reward state
            results[name]['learning_rates'].append(cell.s.alpha_W)
            
            if hasattr(cell.s, 'reward_ema'):
                results[name]['reward_emas'].append(cell.s.reward_ema)
            else:
                results[name]['reward_emas'].append(0.0)
        
        # Progress reporting
        if (step + 1) % 200 == 0 or step + 1 == env.total_steps:
            elapsed = time.time() - start_time
            progress = (step + 1) / env.total_steps
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            
            print(f"  Step {step+1:4d}/{env.total_steps} | "
                  f"Regime: {env.get_regime_params(step)} | "
                  f"ETA: {eta:.1f}s")
            
            # Show current learning rates to see the collapse vs bounded behavior
            current_alphas = {name: cell.s.alpha_W for name, cell in cells.items()}
            alpha_str = " | ".join([f"{name}: {alpha:.2e}" for name, alpha in current_alphas.items()])
            print(f"    α: {alpha_str}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Experiment completed in {total_time:.1f}s")
    
    return results


def analyze_v1_vs_v2_results(results: Dict):
    """Analyze and visualize the comparison results."""
    print(f"\n📊 Analysis: v1 vs v2 Genome Behavior")
    print("=" * 45)
    
    genome_names = [name for name in results.keys() if name != 'environment']
    total_steps = len(results['environment']['rewards'])
    
    # 1) Learning rate analysis - this is the key test
    print(f"\n1️⃣ Learning Rate Analysis (α collapse test)")
    print("-" * 40)
    
    for name in genome_names:
        alphas = results[name]['learning_rates']
        initial_alpha = alphas[0]
        final_alpha = alphas[-1] 
        min_alpha = min(alphas)
        max_alpha = max(alphas)
        
        # Check for collapse (v1) vs bounded (v2)
        collapse_ratio = final_alpha / initial_alpha
        
        print(f"{name}:")
        print(f"  Initial α: {initial_alpha:.2e}")
        print(f"  Final α:   {final_alpha:.2e}")
        print(f"  Range:     [{min_alpha:.2e}, {max_alpha:.2e}]")
        print(f"  Collapse:  {collapse_ratio:.2e}x")
        
        if collapse_ratio < 0.01:  # collapsed by >99%
            print(f"  ❌ COLLAPSED - genome is pathological")
        elif max_alpha / min_alpha > 100:  # unbounded explosion
            print(f"  ⚠️ UNSTABLE - genome shows extreme variation")
        else:
            print(f"  ✅ STABLE - genome maintains bounded learning")
    
    # 2) Performance analysis
    print(f"\n2️⃣ Performance Analysis (prediction error)")
    print("-" * 40)
    
    regime_boundaries = [0, 300, 600, 900, 1200]
    
    for name in genome_names:
        errors = results[name]['errors']
        
        # Overall performance
        mean_error = np.mean(errors)
        final_100_error = np.mean(errors[-100:])  # performance in final regime
        
        print(f"{name}:")
        print(f"  Overall error: {mean_error:.4f}")
        print(f"  Final error:   {final_100_error:.4f}")
        
        # Regime-specific performance
        regime_errors = []
        for i in range(len(regime_boundaries) - 1):
            start_idx = regime_boundaries[i]
            end_idx = regime_boundaries[i + 1]
            regime_error = np.mean(errors[start_idx:end_idx])
            regime_errors.append(regime_error)
            print(f"  Regime {i+1} error: {regime_error:.4f}")
        
        # Adaptation measure (error reduction within regimes)
        adaptations = []
        for i in range(len(regime_boundaries) - 1):
            start_idx = regime_boundaries[i]
            end_idx = regime_boundaries[i + 1]
            regime_slice = errors[start_idx:end_idx]
            
            if len(regime_slice) >= 60:  # enough data for early/late comparison
                early_error = np.mean(regime_slice[:30])
                late_error = np.mean(regime_slice[-30:])
                adaptation = (early_error - late_error) / early_error
                adaptations.append(adaptation)
        
        avg_adaptation = np.mean(adaptations) if adaptations else 0.0
        print(f"  Adaptation score: {avg_adaptation:.3f}")
    
    # 3) Reward EMA tracking (for reward-sensitive genomes)
    print(f"\n3️⃣ Reward Tracking Analysis")
    print("-" * 30)
    
    for name in genome_names:
        if 'reward' in name.lower():
            reward_emas = results[name]['reward_emas']
            if any(r > 0 for r in reward_emas):  # has reward data
                initial_ema = reward_emas[0]
                final_ema = reward_emas[-1]
                mean_ema = np.mean(reward_emas)
                
                print(f"{name}:")
                print(f"  Initial R_ema: {initial_ema:.3f}")
                print(f"  Final R_ema:   {final_ema:.3f}")  
                print(f"  Mean R_ema:    {mean_ema:.3f}")
    
    return {
        'genome_names': genome_names,
        'performance_summary': {
            name: {
                'mean_error': float(np.mean(results[name]['errors'])),
                'final_alpha': results[name]['learning_rates'][-1],
                'alpha_collapse': results[name]['learning_rates'][-1] / results[name]['learning_rates'][0]
            }
            for name in genome_names
        }
    }


def create_comparison_plots(results: Dict, analysis: Dict):
    """Create plots showing v1 vs v2 behavioral differences."""
    genome_names = analysis['genome_names']
    total_steps = len(results['environment']['rewards'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NGISCell v1 vs v2 Genome Comparison', fontsize=16, fontweight='bold')
    
    time_steps = range(total_steps)
    regime_boundaries = [300, 600, 900]
    
    # 1) Learning rate evolution (key plot!)
    ax1 = axes[0, 0]
    ax1.set_title('Learning Rate Evolution (α collapse test)', fontweight='bold')
    
    for name in genome_names:
        alphas = results[name]['learning_rates']
        color = 'red' if 'v1' in name else 'blue' if 'conservative' in name else 'green' if 'aggressive' in name else 'orange'
        linestyle = '--' if 'v1' in name else '-'
        ax1.semilogy(time_steps, alphas, label=name, color=color, linestyle=linestyle, alpha=0.8)
    
    for boundary in regime_boundaries:
        ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Learning Rate (α)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'v1 should collapse\nv2 should stay bounded', 
             transform=ax1.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2) Prediction error
    ax2 = axes[0, 1] 
    ax2.set_title('Prediction Error Over Time')
    
    for name in genome_names:
        errors = results[name]['errors']
        # Smooth with rolling window for readability
        window = 50
        if len(errors) >= window:
            smoothed = np.convolve(errors, np.ones(window)/window, mode='valid')
            steps_smooth = time_steps[window-1:]
            
            color = 'red' if 'v1' in name else 'blue' if 'conservative' in name else 'green' if 'aggressive' in name else 'orange'
            ax2.plot(steps_smooth, smoothed, label=name, color=color, alpha=0.8)
    
    for boundary in regime_boundaries:
        ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Prediction Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3) Reward EMA (for reward-sensitive genomes)
    ax3 = axes[1, 0]
    ax3.set_title('Reward EMA Tracking (reward genomes only)')
    
    reward_genomes = [name for name in genome_names if 'reward' in name.lower()]
    for name in reward_genomes:
        reward_emas = results[name]['reward_emas'] 
        if any(r > 0 for r in reward_emas):
            color = 'red' if 'v1' in name else 'blue' if 'conservative' in name else 'green' if 'aggressive' in name else 'orange'
            linestyle = '--' if 'v1' in name else '-'
            ax3.plot(time_steps, reward_emas, label=name, color=color, linestyle=linestyle, alpha=0.8)
    
    # Also plot true reward signal
    env_rewards = results['environment']['rewards']
    ax3.plot(time_steps, env_rewards, label='True Reward', color='black', alpha=0.3, linewidth=0.5)
    
    for boundary in regime_boundaries:
        ax3.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reward EMA') 
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4) Final performance comparison
    ax4 = axes[1, 1]
    ax4.set_title('Final Performance Summary')
    
    performance_data = analysis['performance_summary']
    names = list(performance_data.keys())
    mean_errors = [performance_data[name]['mean_error'] for name in names]
    collapse_ratios = [performance_data[name]['alpha_collapse'] for name in names]
    
    # Color by genome type
    colors = ['red' if 'v1' in name else 'blue' if 'conservative' in name else 'green' if 'aggressive' in name else 'orange' for name in names]
    
    scatter = ax4.scatter(mean_errors, collapse_ratios, c=colors, s=100, alpha=0.7)
    
    for i, name in enumerate(names):
        ax4.annotate(name.replace('_', '\n'), (mean_errors[i], collapse_ratios[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Mean Prediction Error')
    ax4.set_ylabel('α Collapse Ratio (final/initial)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
    
    plt.tight_layout()
    plt.savefig('ngis_v1_vs_v2_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved: ngis_v1_vs_v2_comparison.png")
    
    return fig


def main():
    """Run the complete v1 vs v2 comparison experiment."""
    # Run experiment
    results = run_v1_vs_v2_experiment()
    
    # Analyze results
    analysis = analyze_v1_vs_v2_results(results)
    
    # Create plots
    fig = create_comparison_plots(results, analysis)
    
    # Save results
    results_file = 'ngis_v1_vs_v2_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved: {results_file}")
    
    # Summary conclusion
    print(f"\n🎯 CONCLUSION: v1 vs v2 Genome Comparison")
    print("=" * 45)
    
    performance = analysis['performance_summary']
    
    # Check for v1 collapse
    v1_collapsed = False
    v2_stable = True
    
    for name, stats in performance.items():
        if 'v1' in name and stats['alpha_collapse'] < 0.01:
            v1_collapsed = True
            print(f"✅ {name}: COLLAPSED as expected ({stats['alpha_collapse']:.2e}x)")
        elif 'v2' in name:
            if 0.01 <= stats['alpha_collapse'] <= 100:
                print(f"✅ {name}: STABLE learning rate ({stats['alpha_collapse']:.2e}x)")
            else:
                v2_stable = False
                print(f"❌ {name}: UNSTABLE ({stats['alpha_collapse']:.2e}x)")
    
    if v1_collapsed and v2_stable:
        print(f"\n🎉 SUCCESS: v2 genome architecture FIXES the α→0 collapse!")
        print(f"   - v1 genomes collapse as expected (pathological)")  
        print(f"   - v2 genomes maintain bounded, stable learning")
        print(f"   - Genome-driven behavior differences preserved")
    else:
        print(f"\n⚠️ Mixed results - need further investigation")
    
    # Show best performer
    best_name = min(performance.keys(), key=lambda x: performance[x]['mean_error'])
    best_error = performance[best_name]['mean_error']
    print(f"\n🏆 Best performer: {best_name} (error: {best_error:.4f})")


if __name__ == "__main__":
    main()