"""
Robust Multivector implementation for Cl(3,0) geometric algebra.
Built for numerical stability and performance in NGIS v0.
"""
import numpy as np
import math
from typing import Union, Tuple


class Multivector:
    """
    Cl(3,0) multivector with 8 components: [α, a1, a2, a3, b12, b23, b31, β]
    
    Components:
    - α: scalar
    - a1, a2, a3: vector components (e1, e2, e3)  
    - b12, b23, b31: bivector components (e1∧e2, e2∧e3, e3∧e1)
    - β: trivector/pseudoscalar (e1∧e2∧e3)
    """
    
    def __init__(self, coeffs: Union[list, tuple, np.ndarray]):
        if len(coeffs) != 8:
            raise ValueError(f"Multivector requires 8 coefficients, got {len(coeffs)}")
        self.c = np.array(coeffs, dtype=np.float64)
    
    @classmethod
    def zero(cls):
        """Create zero multivector."""
        return cls([0.0] * 8)
    
    @classmethod
    def scalar(cls, value: float):
        """Create scalar multivector."""
        return cls([value, 0, 0, 0, 0, 0, 0, 0])
    
    @classmethod
    def vector(cls, x: float, y: float, z: float):
        """Create vector multivector."""
        return cls([0, x, y, z, 0, 0, 0, 0])
    
    @classmethod
    def from_vector3(cls, v: np.ndarray):
        """Create vector multivector from 3D numpy array."""
        return cls([0, v[0], v[1], v[2], 0, 0, 0, 0])
    
    def __add__(self, other):
        """Multivector addition."""
        if isinstance(other, Multivector):
            return Multivector(self.c + other.c)
        elif isinstance(other, (int, float)):
            result = self.c.copy()
            result[0] += other  # Add to scalar part
            return Multivector(result)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """Multivector subtraction."""
        if isinstance(other, Multivector):
            return Multivector(self.c - other.c)
        elif isinstance(other, (int, float)):
            result = self.c.copy()
            result[0] -= other
            return Multivector(result)
        else:
            return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result = -self.c.copy()
            result[0] += other
            return Multivector(result)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        """Scalar multiplication or geometric product."""
        if isinstance(other, (int, float)):
            return Multivector(self.c * other)
        elif isinstance(other, Multivector):
            return self.geometric_product(other)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector(self.c * other)
        else:
            return NotImplemented
    
    def __truediv__(self, scalar):
        """Scalar division."""
        if isinstance(scalar, (int, float)):
            if abs(scalar) < 1e-14:
                raise ValueError("Division by zero")
            return Multivector(self.c / scalar)
        else:
            return NotImplemented
    
    def __neg__(self):
        """Negation."""
        return Multivector(-self.c)
    
    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """
        Compute geometric product using correct Cl(3,0) multiplication table.
        
        Basis ordering: [1, e1, e2, e3, e12, e23, e31, e123]
        Index mapping: [0, 1,  2,  3,  4,   5,   6,   7  ]
        
        Key relations:
        e1*e1 = e2*e2 = e3*e3 = 1
        e1*e2 = e12, e2*e1 = -e12
        e2*e3 = e23, e3*e2 = -e23  
        e3*e1 = e31, e1*e3 = -e31
        e12*e3 = e123, e3*e12 = -e123
        e23*e1 = e123, e1*e23 = -e123
        e31*e2 = e123, e2*e31 = -e123
        """
        a, b = self.c, other.c
        result = np.zeros(8)
        
        # Grade 0 (scalar)
        result[0] = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] - 
                    a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7])
        
        # Grade 1 (vector)  
        result[1] = (a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[6] + 
                    a[4]*b[2] - a[5]*b[7] - a[6]*b[3] - a[7]*b[5])
        result[2] = (a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[5] - 
                    a[4]*b[1] + a[5]*b[3] - a[6]*b[7] - a[7]*b[6])
        result[3] = (a[0]*b[3] - a[1]*b[6] + a[2]*b[5] + a[3]*b[0] + 
                    a[4]*b[7] - a[5]*b[2] + a[6]*b[1] - a[7]*b[4])
        
        # Grade 2 (bivector)
        result[4] = (a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7] + 
                    a[4]*b[0] + a[5]*b[6] - a[6]*b[5] + a[7]*b[3])
        result[5] = (a[0]*b[5] - a[1]*b[7] + a[2]*b[3] - a[3]*b[2] - 
                    a[4]*b[6] + a[5]*b[0] + a[6]*b[4] + a[7]*b[1])
        result[6] = (a[0]*b[6] + a[1]*b[3] - a[2]*b[7] + a[3]*b[1] + 
                    a[4]*b[5] - a[5]*b[4] + a[6]*b[0] + a[7]*b[2])
        
        # Grade 3 (pseudoscalar)
        result[7] = (a[0]*b[7] + a[1]*b[5] + a[2]*b[6] + a[3]*b[4] + 
                    a[4]*b[3] + a[5]*b[1] + a[6]*b[2] + a[7]*b[0])
        
        return Multivector(result)
    
    def reverse(self) -> 'Multivector':
        """
        Compute reverse (conjugate) of multivector.
        Reverses sign of bivector and pseudoscalar parts.
        """
        result = self.c.copy()
        result[4:8] = -result[4:8]  # Negate bivector and pseudoscalar
        return Multivector(result)
    
    def magnitude_squared(self) -> float:
        """Compute squared magnitude using scalar part of M * M.reverse()."""
        rev = self.reverse()
        product = self.geometric_product(rev)
        return product.c[0]  # Scalar part
    
    def magnitude(self) -> float:
        """Compute magnitude with numerical stability."""
        mag_sq = self.magnitude_squared()
        if mag_sq < 0:
            # Should not happen in Cl(3,0) but guard against numerical issues
            return 0.0
        return math.sqrt(mag_sq)
    
    def normalize(self, eps: float = 1e-12) -> 'Multivector':
        """
        Normalize multivector to unit magnitude.
        Returns zero multivector if magnitude is below epsilon.
        """
        mag = self.magnitude()
        if mag < eps:
            return Multivector.zero()
        return self / mag
    
    def get_scalar(self) -> float:
        """Get scalar part."""
        return self.c[0]
    
    def get_vector(self) -> np.ndarray:
        """Get vector part as 3D numpy array."""
        return self.c[1:4].copy()
    
    def get_bivector(self) -> np.ndarray:
        """Get bivector part."""
        return self.c[4:7].copy()
    
    def get_pseudoscalar(self) -> float:
        """Get pseudoscalar part."""
        return self.c[7]
    
    def vector_magnitude(self) -> float:
        """Get magnitude of vector part only."""
        return np.linalg.norm(self.c[1:4])
    
    def is_zero(self, eps: float = 1e-12) -> bool:
        """Check if multivector is approximately zero."""
        return np.allclose(self.c, 0, atol=eps)
    
    def is_scalar(self, eps: float = 1e-12) -> bool:
        """Check if multivector is purely scalar."""
        return np.allclose(self.c[1:], 0, atol=eps)
    
    def is_vector(self, eps: float = 1e-12) -> bool:
        """Check if multivector is purely vector."""
        non_vector = np.concatenate([self.c[0:1], self.c[4:]])
        return np.allclose(non_vector, 0, atol=eps)
    
    def __str__(self) -> str:
        """String representation."""
        parts = []
        labels = ['', 'e1', 'e2', 'e3', 'e12', 'e23', 'e31', 'e123']
        
        for i, coeff in enumerate(self.c):
            if abs(coeff) > 1e-12:
                if i == 0:
                    parts.append(f"{coeff:.6f}")
                else:
                    parts.append(f"{coeff:.6f}{labels[i]}")
        
        if not parts:
            return "0"
        return " + ".join(parts).replace("+ -", "- ")
    
    def __repr__(self) -> str:
        return f"Multivector({self.c.tolist()})"
    
    def copy(self) -> 'Multivector':
        """Create a copy of this multivector."""
        return Multivector(self.c.copy())


def rotor_from_vectors(v1: np.ndarray, v2: np.ndarray, angle_factor: float = 1.0, 
                      eps: float = 1e-12) -> Tuple[Multivector, float]:
    """
    Construct rotor to rotate from v1 toward v2 by angle_factor * angle_between.
    
    Args:
        v1: Source vector (3D numpy array)
        v2: Target vector (3D numpy array)  
        angle_factor: Fraction of total angle to rotate (0 to 1)
        eps: Numerical epsilon for stability
    
    Returns:
        Tuple of (rotor_multivector, actual_angle_rotated)
    """
    # Normalize input vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < eps or v2_norm < eps:
        # Degenerate case - return identity rotor
        return Multivector.scalar(1.0), 0.0
    
    v1_hat = v1 / v1_norm
    v2_hat = v2 / v2_norm
    
    # Compute dot product and angle
    dot = np.clip(np.dot(v1_hat, v2_hat), -1.0, 1.0)
    
    # Handle aligned vectors
    if abs(dot - 1.0) < eps:
        return Multivector.scalar(1.0), 0.0  # Already aligned
    
    # Handle anti-aligned vectors
    if abs(dot + 1.0) < eps:
        # 180 degree rotation - choose perpendicular plane
        # Find a vector orthogonal to v1_hat
        if abs(v1_hat[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        
        ortho = perp - np.dot(perp, v1_hat) * v1_hat
        ortho = ortho / np.linalg.norm(ortho)
        
        # Create bivector for 180 degree rotation
        cross = np.cross(v1_hat, ortho)
        bivector_coeffs = [cross[0], cross[1], cross[2]]  # e12, e23, e31 components
        
        # Half-angle for 180 degrees is 90 degrees
        half_angle = angle_factor * math.pi / 2.0
        cos_half = math.cos(half_angle)
        sin_half = math.sin(half_angle)
        
        rotor = Multivector([cos_half, 0, 0, 0, 
                           -sin_half * bivector_coeffs[0],
                           -sin_half * bivector_coeffs[1], 
                           -sin_half * bivector_coeffs[2], 0])
        
        return rotor, angle_factor * math.pi
    
    # General case
    angle = math.acos(abs(dot))
    
    # Compute bivector (rotation plane)
    cross = np.cross(v1_hat, v2_hat)
    cross_mag = np.linalg.norm(cross)
    
    if cross_mag < eps:
        # Vectors are parallel
        return Multivector.scalar(1.0), 0.0
    
    cross_hat = cross / cross_mag
    
    # Scale angle by factor
    scaled_angle = angle_factor * angle
    half_angle = scaled_angle / 2.0
    
    cos_half = math.cos(half_angle)
    sin_half = math.sin(half_angle)
    
    # Construct rotor: R = cos(θ/2) - B̂ sin(θ/2)
    # Map cross product to bivector basis: cross = [cx, cy, cz] -> [e23, e31, e12]  
    # This is because e1×e2=e3 -> e12, e2×e3=e1 -> e23, e3×e1=e2 -> e31
    rotor = Multivector([
        cos_half,
        0, 0, 0,
        -sin_half * cross_hat[2],  # e12 component (from z cross product)
        -sin_half * cross_hat[0],  # e23 component (from x cross product)
        -sin_half * cross_hat[1],  # e31 component (from y cross product)  
        0
    ])
    
    return rotor, scaled_angle


def apply_rotor(rotor: Multivector, target: Multivector) -> Multivector:
    """
    Apply rotor to multivector: result = R * target * R†
    """
    rotor_rev = rotor.reverse()
    temp = rotor.geometric_product(target)
    result = temp.geometric_product(rotor_rev)
    return result


if __name__ == "__main__":
    # Basic validation tests
    print("Testing Multivector implementation...")
    
    # Test basic operations
    mv1 = Multivector.vector(1, 0, 0)
    mv2 = Multivector.vector(0, 1, 0)
    
    print(f"mv1 = {mv1}")
    print(f"mv2 = {mv2}")
    
    # Test geometric product
    product = mv1 * mv2
    print(f"mv1 * mv2 = {product}")
    
    # Test rotor construction
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    
    rotor, angle = rotor_from_vectors(v1, v2, 0.5)  # Rotate halfway
    print(f"Rotor for 45° rotation: {rotor}")
    print(f"Actual angle: {math.degrees(angle)}°")
    
    # Apply rotor
    mv1_rotated = apply_rotor(rotor, mv1)
    print(f"Rotated vector: {mv1_rotated}")
    print(f"Expected ~(0.707, 0.707, 0): {mv1_rotated.get_vector()}")
    
    print("\nMultivector implementation complete!")