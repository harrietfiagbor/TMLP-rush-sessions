import numpy as np
from scipy import spatial

# cosine_similarity(A, B) = (A . B) / (|A| * |B|)
# cosine similarity = dot product / (magnitude of A * magnitude of B)

# dot product
A = np.array([2, 3, 4])
B = np.array([1, 2, 3])
dot_product = np.dot(A, B)

# magnitude
magnitude = np.linalg.norm(A) * np.linalg.norm(B)

cosine_similarity_1 = dot_product / magnitude
print(cosine_similarity_1)
# 0.9838699100999074

# using scipy: the scientific python library
cosine_similarity_2 = -1 * (spatial.distance.cosine(A, B) - 1)
print(cosine_similarity_2)
# 0.9838699100999074
