import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 2, dpi=100, figsize=(6, 6.5))

photo1 = plt.imread("f-orbitals_C3.png")
photo2 = plt.imread("f-orbitals_C2.png")
photo3 = plt.imread("f-orbitals_S4.png")
photo4 = plt.imread("f-orbitals_sigma.png")

ax[0, 0].imshow(photo1)
ax[0, 0].set_axis_off()
ax[0, 0].set_title(r"class $\{C_3\}$", y=-0.12)

ax[0, 1].imshow(photo2)
ax[0, 1].set_axis_off()
ax[0, 1].set_title(r"class $\{C_2\}$", y=-0.11)

ax[1, 0].imshow(photo3)
ax[1, 0].set_axis_off()
ax[1, 0].set_title(r"class $\{S_4\}$", y=-0.1)

ax[1, 1].imshow(photo4)
ax[1, 1].set_axis_off()
ax[1, 1].set_title(r"class $\{\sigma_{\rm d}\}$", y=-0.22)

#fig.subplots_adjust(bottom=0.1, top=0.9, hspace=2, wspace=0)

#fig.suptitle("aaa", y=0.05, va="top")

plt.tight_layout(w_pad=0, h_pad=0.5)

fig.savefig('T_d.png', transparent=True)