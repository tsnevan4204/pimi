import numpy as np
import matplotlib.pyplot as plt

# load vector
vec = np.loadtxt("pc1.txt")

# reshape into image
img = vec.reshape(64, 64)

# normalize for display
img = (img - img.min()) / (img.max() - img.min())

plt.imshow(img, cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()