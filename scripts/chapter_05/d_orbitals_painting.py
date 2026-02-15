import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def get_real_orbital(l, m, theta, phi):
    if m > 0:
        return (1/np.sqrt(2)) * (sph_harm(m, l, phi, theta) + ((-1)**m) * sph_harm(-m, l, phi, theta))
    elif m < 0:
        return (1/(np.sqrt(2)*1j)) * (sph_harm(abs(m), l, phi, theta) - ((-1)**abs(m)) * sph_harm(-abs(m), l, phi, theta))
    return sph_harm(0, l, phi, theta)

def draw_axes(ax, length=0.6):
    ax.quiver(0, 0, 0, length, 0, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, length, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, length, color='black', arrow_length_ratio=0.1)
    
    offset = 0.05
    ax.text(length + offset, 0, 0, "x", color='red', ha='center', va='center', fontsize=12)
    ax.text(0, length + offset, 0, "y", color='green', ha='center', va='center', fontsize=12)
    ax.text(0, 0, length + offset, "z", color='blue', ha='center', va='center', fontsize=12)

def plot_individual_orbitals(orbitals):
    t = np.linspace(0, np.pi, 100)
    p = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(t, p)

    for l, m, name in orbitals:
        fig = plt.figure(figsize=(6, 6), dpi=100) 
        
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        
        y_lm = get_real_orbital(l, m, theta, phi)
        r = np.abs(y_lm)
        x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        
        fcolors = np.real(y_lm)
        colors = plt.cm.RdBu_r((fcolors - fcolors.min()) / (fcolors.max() - fcolors.min() + 1e-9))
        
        ax.plot_surface(x, y, z, facecolors=colors, alpha=0.9, linewidth=0, antialiased=True)
        
        draw_axes(ax, length=0.75)
        limit = 0.5
        ax.set_xlim([-limit, limit]); ax.set_ylim([-limit, limit]); ax.set_zlim([-limit, limit])
        
        ax.axis('off')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)

        clean_name = name.replace('$', '').replace('\\', '')
        filename = f"orbital_{clean_name}.png"
        
        # 强制保存为 600x600 的一种技巧：先 tight 再设置
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1, transparent=True)
        print(f"已生成紧凑版图片: {filename}")
        plt.show()
        plt.close(fig)

# 绘制 3d 轨道 (按常用的实函数顺序)
# m=2: x2-y2, m=-2: xy, m=1: xz, m=-1: yz, m=0: z2
d_orbitals = [
    (2, -2, "$d_{xy}$"), 
    (2, -1, "$d_{yz}$"), 
    (2, 0, "$d_{z^2}$"), 
    (2, 1, "$d_{xz}$"), 
    (2, 2, "$d_{x^2-y^2}$")
]

# 调用同样的函数绘制 d 轨道
plot_individual_orbitals(d_orbitals)