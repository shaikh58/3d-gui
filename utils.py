import numpy as np

def plane_from_points(P1, P2, P3):
    # Convert to numpy arrays
    P1, P2, P3 = map(np.array, (P1, P2, P3))
    
    # Two vectors in the plane
    v1 = P2 - P1
    v2 = P3 - P1

    # Normal vector
    normal = np.cross(v1, v2)
    A, B, C = normal

    # Plane equation: Ax + By + Cz + D = 0
    D = -np.dot(normal, P1)

    return A, B, C, D  # coefficients of the plane

def line_from_points(P1, P2):
    # Convert to numpy arrays
    P1, P2 = map(np.array, (P1, P2))
    
    # Two vectors in the plane
    v1 = P2 - P1
    A, B, C = v1

    return A, B, C  # coefficients of the line