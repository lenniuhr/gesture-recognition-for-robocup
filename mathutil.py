import math
import numpy as np

# Normalize np vector
def normalize(v1):
    v1_length = np.linalg.norm(v1)
    return v1 / v1_length

# Calculate angle between 2 np vectors
def winkel(v1, v2):
	v1_norm = normalize(v1)
	v2_norm = normalize(v2)

	product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
	product = max(-1, min(product, 1))

	acos = math.acos(product)
	winkel = math.degrees(acos)
	return winkel