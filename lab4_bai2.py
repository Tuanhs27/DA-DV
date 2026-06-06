import numpy as np
import matplotlib.pyplot as plt

values = np.random.rand(100)

size = int(np.ceil(np.sqrt(len(values))))
pixel_matrix = np.zeros(size * size)
pixel_matrix[:len(values)] = values
pixel_matrix = pixel_matrix.reshape(size, size)

plt.figure(figsize=(6, 6))
plt.imshow(pixel_matrix, cmap="viridis")
plt.colorbar()
plt.title("Pixel-based Visualization")
plt.show()