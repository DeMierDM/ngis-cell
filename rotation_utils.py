"""
Rotation Utilities for Regime-Based Environment
=============================================

Helper functions for applying rotations in the 4-regime environment.
Uses geometric algebra (GA) rotors for consistent 3D rotations.
"""

import numpy as np
import math
from multivector import Multivector


def apply_rotation(v: np.ndarray, axis: np.ndarray, omega: float) -> np.ndarray:
    """
    Apply rotation to vector v around axis by angle omega (radians).
    
    Args:
        v: 3D vector to rotate
        axis: 3D unit vector (rotation axis)  
        omega: rotation angle in radians
        
    Returns:
        rotated 3D vector
    """
    # Normalize inputs
    v = v / (np.linalg.norm(v) + 1e-8)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Create rotor using GA: R = cos(θ/2) + sin(θ/2) * B
    # where B is the bivector (axis in GA)
    half_angle = omega / 2.0
    cos_half = math.cos(half_angle)
    sin_half = math.sin(half_angle)
    
    # GA rotor coefficients: [scalar, e1, e2, e3, e23, e13, e12, e123]
    rotor_coeffs = np.zeros(8)
    rotor_coeffs[0] = cos_half  # scalar part
    rotor_coeffs[4] = sin_half * axis[2]  # e23 component (x-axis rotation)
    rotor_coeffs[5] = sin_half * axis[1]  # e13 component (y-axis rotation) 
    rotor_coeffs[6] = sin_half * axis[0]  # e12 component (z-axis rotation)
    
    rotor = Multivector(rotor_coeffs)
    
    # Convert vector to GA multivector
    v_mv = Multivector.from_vector3(v)
    
    # Apply rotation: v' = R * v * R†
    try:
        from multivector import apply_rotor
        rotated_mv = apply_rotor(rotor, v_mv)
        return rotated_mv.c[1:4]  # Extract vector part
    except ImportError:
        # Fallback: manual rotor application
        # R† (rotor conjugate)
        rotor_conj_coeffs = rotor.c.copy()
        rotor_conj_coeffs[4:7] *= -1  # Negate bivector parts
        rotor_conj = Multivector(rotor_conj_coeffs)
        
        # R * v * R†
        temp = rotor * v_mv
        rotated_mv = temp * rotor_conj
        
        return rotated_mv.c[1:4]  # Extract vector part


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Safely normalize a 3D vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return v / norm


def random_unit_vector() -> np.ndarray:
    """Generate random unit vector using Gaussian method."""
    v = np.random.normal(size=3)
    return normalize_vector(v)


def rodrigues_rotation(v: np.ndarray, axis: np.ndarray, omega: float) -> np.ndarray:
    """
    Alternative rotation using Rodrigues' formula (fallback).
    v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    """
    axis = normalize_vector(axis)
    cos_omega = math.cos(omega)
    sin_omega = math.sin(omega)
    
    # k×v (cross product)
    k_cross_v = np.cross(axis, v)
    
    # k·v (dot product)
    k_dot_v = np.dot(axis, v)
    
    # Rodrigues formula
    v_rotated = (
        v * cos_omega + 
        k_cross_v * sin_omega + 
        axis * k_dot_v * (1.0 - cos_omega)
    )
    
    return v_rotated


if __name__ == "__main__":
    print("🔄 Testing Rotation Utilities")
    print("=" * 30)
    
    # Test basic rotation
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # z-axis
    omega = math.pi / 2  # 90 degrees
    
    print(f"Original vector: {v}")
    print(f"Rotation axis: {axis}")
    print(f"Rotation angle: {omega:.3f} rad ({math.degrees(omega):.1f}°)")
    
    # Test GA rotation
    try:
        v_rotated = apply_rotation(v, axis, omega)
        print(f"GA rotated: {v_rotated}")
        print(f"Expected: [0, 1, 0] (approx)")
    except Exception as e:
        print(f"GA rotation failed: {e}")
        
    # Test Rodrigues rotation
    v_rodrigues = rodrigues_rotation(v, axis, omega)
    print(f"Rodrigues rotated: {v_rodrigues}")
    
    # Test random unit vector
    random_v = random_unit_vector()
    print(f"Random unit vector: {random_v} (norm: {np.linalg.norm(random_v):.6f})")
    
    print("✅ Rotation utilities ready for regime environment")