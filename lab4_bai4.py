import numpy as np
import matplotlib.pyplot as plt

def draw_face(ax, data):
    face_size = 0.5 + data[0] * 0.5
    eye_size = 0.05 + data[1] * 0.05
    mouth_curve = data[2] - 0.5
    nose_size = 0.05 + data[3] * 0.05
    
    face = plt.Circle((0.5, 0.5), face_size * 0.4, fill=False, linewidth=2)
    ax.add_patch(face)
    
    left_eye = plt.Circle((0.35, 0.6), eye_size, color="black")
    right_eye = plt.Circle((0.65, 0.6), eye_size, color="black")
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    nose = plt.Circle((0.5, 0.5), nose_size, color="black")
    ax.add_patch(nose)
    
    x = np.linspace(0.35, 0.65, 100)
    y = 0.35 + mouth_curve * (x - 0.5)**2 * -4
    ax.plot(x, y, linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

samples = np.random.rand(4, 4)
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    draw_face(ax, samples[i])
    ax.set_title(f"Sample {i+1}")

plt.tight_layout()
plt.show()