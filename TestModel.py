import numpy as np
import matplotlib.pyplot as plt
from TrainingModel import particle_filter, maximization_step

# Đọc mô hình huấn luyện đã lưu
particles = np.load('particles.npy')
weights = np.load('weights.npy')

# Khởi tạo robot và các tham số
state = np.array([0.0, 0.0, 0.0])  # Vị trí ban đầu của robot (x, y, yaw)
u = np.array([1.0, 0.1])  # Điều khiển (v, w)

# Áp dụng Particle Filter cho thử nghiệm
particles, weights = particle_filter(state, particles, weights, u)

# Hiển thị kết quả
plt.scatter(particles[:, 0], particles[:, 1], c=weights, cmap="viridis", alpha=0.6)
plt.plot(state[0], state[1], "ro", label="Robot Position")
plt.title("Robot Position and Particles after Particle Filter")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()