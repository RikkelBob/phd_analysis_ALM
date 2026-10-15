import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import config

# --- USER INPUT ---
data_dir = config.DATA_DIR_PP / "light_spread"
config_file = os.path.join(data_dir, "lambda_fiber_light_spread_aux.txt")
data_file = os.path.join(data_dir, "lambda_fiber_light_spread.txt")
# ------------------

# --- Parse config ---
config = {}
with open(config_file) as f:
    for line in f:
        line = line.strip()
        if '=' in line:
            key, _, val = line.partition('=')
            config[key.strip()] = val.strip()

rMAX = float(config['rMAX'])
zrange = float(config['zrange'])
delta = float(config['Delta'])
EL = float(config['EL'])

# --- Load ---
grid = np.loadtxt(data_file)
nz, nx = grid.shape
print(f"Grid loaded: {nz} rows (z) x {nx} cols (x)")

x = np.linspace(-rMAX, rMAX, nx)
y = np.linspace(-2, 4, nz)

# --- Sample at x = ±0.5 and ±1.0, at y = 0.5 and y = 1.0 ---
print("\n--- Intensity samples ---")
print(f"{'y':>8} {'x':>8} {'intensity':>14}")
print("-" * 34)
for yd in [0.5, 1.0]:
    iz = np.argmin(np.abs(y - yd))
    for xd in [-1.0, -0.5, 0.5, 1.0]:
        ix = np.argmin(np.abs(x - xd))
        print(f"{yd:>8.1f} {xd:>+8.1f} {grid[iz, ix]:>14.4e}")

# --- Plot (log scale) ---
fig, ax = plt.subplots(figsize=(6, 8))

vmin = grid[grid > 0].min()
vmax = grid.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

im = ax.pcolormesh(x, y, grid, shading='auto', cmap='inferno', norm=norm)
plt.colorbar(im, ax=ax, label='Intensity (log scale)')

# 99% boundary
threshold = vmax * 0.01
contour = ax.contour(x, y, grid, levels=[threshold], colors='cyan', linewidths=1.5)

# Max x extent of boundary on each side
above = grid >= threshold
cols_above = np.any(above, axis=0)
x_pos_max = x[cols_above & (x >= 0)].max()
x_neg_max = x[cols_above & (x <= 0)].min()
print(f"\n99% boundary x extent:  positive: +{x_pos_max:.3f} mm  |  negative: {x_neg_max:.3f} mm")

ax.set_xlim(-rMAX, rMAX)
ax.set_ylim(-2, 4)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('Light Distribution (log scale)')

plt.tight_layout()
plt.savefig('light_distribution.png', dpi=150)
plt.show()

print(f"\nPeak intensity:         {vmax:.4e}")
print(f"99% boundary threshold: {threshold:.4e}")
