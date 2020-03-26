from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

mean_2d = [0, 0]
cov_2d = [[1,0.8],[0.8,1]]

mean2_2d = [0, 0]
cov2_2d = [[1,-0.8],[-0.8,1]]

x_2d, y_2d = np.random.multivariate_normal(mean_2d, cov_2d, 5000).T
x2_2d, y2_2d = np.random.multivariate_normal(mean2_2d, cov2_2d, 5000).T

mean = [0, 0, 0]
cov = [[0.1,0,0],[0,10,0],[0,0,10]]

mean2 = [0, 0, 0]
cov2 = [[10,0,0],[0,0.1,0],[0,0,10]]

mean3 = [0, 0, 0]
cov3 = [[10,0,0],[0,10,0],[0,0,0.1]]

x, y, z = np.random.multivariate_normal(mean, cov, 5000).T
x2, y2, z2 = np.random.multivariate_normal(mean2, cov2, 5000).T
x3, y3, z3 = np.random.multivariate_normal(mean3, cov3, 5000).T

fig = plt.figure()
ax_2d = fig.add_subplot(121)
ax = fig.add_subplot(122, projection="3d")

ax_2d.set_xlim([-2,2])
ax_2d.set_ylim([-2,2])
ax_2d.scatter(x_2d, y_2d, alpha=0.2)
ax_2d.scatter(x2_2d, y2_2d, alpha=0.2)
leg = ax_2d.legend(["class 0", "class 1"], fancybox=True, framealpha=1)
for lh in leg.legendHandles:
  lh.set_alpha(1)
ax_2d.set_xlabel("$x_1$")
ax_2d.set_ylabel("$x_2$")
#ax_2d.text(-0.10, -2.5, "(a)")
ax_2d.set_title("(a)")

ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_zlim([-10,10])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.scatter(x, y, z, alpha=0.2)
ax.scatter(x2, y2, z2, alpha=0.2)
ax.scatter(x3, y3, z3, alpha=0.2)
leg = ax.legend(["class 0", "class 1", "class 1"], fancybox=True, framealpha=1)
for lh in leg.legendHandles:
  lh.set_alpha(1)
# ax.dist = 8
ax.dist = 6
#ax.text(-0.10, -5, -5, "(b)")
ax.set_title("(b)")


mean_2d = [str(i) for i in mean_2d]
mean2_2d = [str(i) for i in mean2_2d]

cov_2d = [[str(j) for j in i] for i in cov_2d]
cov2_2d = [[str(j) for j in i] for i in cov2_2d]

filename_2d = "data-2d-" + "#".join(mean_2d) + "-" + "#".join(["#".join(i) for i in cov_2d]) + ".npy"
filename2_2d = "data-2d-" + "#".join(mean2_2d) + "-" + "#".join(["#".join(i) for i in cov2_2d]) + ".npy"

# np.save(filename_2d, np.stack([x_2d,y_2d], axis=1))
# np.save(filename2_2d, np.stack([x2_2d,y2_2d], axis=1))

# np.save("data-2d-clsa.npy", np.stack([x_2d,y_2d], axis=1))
# np.save("data-2d-clsb.npy", np.stack([x2_2d,y2_2d], axis=1))

mean = [str(i) for i in mean]
mean2 = [str(i) for i in mean2]
mean3 = [str(i) for i in mean3]

cov = [[str(j) for j in i] for i in cov]
cov2 = [[str(j) for j in i] for i in cov2]
cov3 = [[str(j) for j in i] for i in cov3]

filename = "data-3d-" + "#".join(mean) + "-" + "#".join(["#".join(i) for i in cov]) + ".npy"
filename2 = "data-3d-" + "#".join(mean2) + "-" + "#".join(["#".join(i) for i in cov2]) + ".npy"
filename3 = "data-3d-" + "#".join(mean3) + "-" + "#".join(["#".join(i) for i in cov3]) + ".npy"

# np.save(filename, np.stack([x,y,z], axis=1))
# np.save(filename2, np.stack([x2,y2,z2], axis=1))
# np.save(filename3, np.stack([x2,y2,z3], axis=1))

# np.save("data-3d-clsa.npy", np.stack([x,y,z], axis=1))
# np.save("data-3d-clsb.npy", np.stack([x2,y2,z2], axis=1))
# np.save("data-3d-clsc.npy", np.stack([x2,y2,z3], axis=1))


plt.subplots_adjust(bottom=0.3, left=0.07, right=0.85)
plt.show()
