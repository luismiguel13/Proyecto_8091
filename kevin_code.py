import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging

parser = argparse.ArgumentParser('Seismic project', add_help=False)
parser.add_argument('--data_folder', default='./data', type=str, help='Folder path to dataset')
parser.add_argument('--magnetometry_folder', default='Datos_Magnetometria_Anomalia_Magnetica_de_Campo_Total', type=str, help='Folder path to magnetometry data')
parser.add_argument('--gravimetry_folder', default='Datos_Gravimetria_Anomalia_Residual', type=str, help='Folder path to gravimetry data')
args = parser.parse_args()

## load data
magnetometry_path = os.path.join(args.data_folder, args.magnetometry_folder, "Puntos_Grilla_Anomalia_Magnetica_de_Campo_Total_SGC.xlsx")
gravimetry_path = os.path.join(args.data_folder, args.gravimetry_folder, "Puntos_Grilla_Anomalia_Residual_SGC.xlsx")

df_magnetometry = pd.read_excel(magnetometry_path)
df_gravimetry = pd.read_excel(gravimetry_path)

## Visualization
# Scatter plot of residual gravity anomaly
plt.scatter(df_gravimetry["Este"], df_gravimetry["Norte"], c=df_gravimetry["Anomalia_Residual"], cmap="seismic")
plt.colorbar(label="Residual Gravity (mGal)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Gravity Residual Anomaly Map")
plt.show()

# Scatter plot of magnetic anomaly
plt.scatter(df_magnetometry["Este"], df_magnetometry["Norte"], c=df_magnetometry["Anomalia_Magnetica_Campo_Total"], cmap="viridis")
plt.colorbar(label="Magnetic Anomaly (nT)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Magnetic Total Field Map")
plt.show()

## Interpolations

# Coordinates and values
x_grav = df_gravimetry["Este"].values
y_grav = df_gravimetry["Norte"].values
z_grav = df_gravimetry["Anomalia_Residual"].values

x_magn = df_magnetometry["Este"].values
y_magn = df_magnetometry["Norte"].values
z_magn = df_magnetometry["Anomalia_Magnetica_Campo_Total"].values


## Interpolation method (1): Inverse Distance Weighting
# Define output grid
grid_x_gravm1, grid_y_gravm1 = np.mgrid[x_grav.min():x_grav.max():200j, y_grav.min():y_grav.max():200j]

grid_z_gravm1 = griddata((x_grav, y_grav), z_grav, (grid_x_gravm1, grid_y_gravm1), method="cubic")  # method: 'linear', 'cubic', or 'nearest'

plt.figure(figsize=(8, 6))
plt.contourf(grid_x_gravm1, grid_y_gravm1, grid_z_gravm1, 30, cmap="seismic")
plt.colorbar(label="Residual Gravity (mGal)")
plt.scatter(x_grav, y_grav, c="k", s=10, label="Data points")
plt.legend()
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Interpolated Gravity Anomaly")
plt.show()

# Define output grid
grid_x_magn1, grid_y_magn1 = np.mgrid[x_magn.min():x_magn.max():200j, y_magn.min():y_magn.max():200j]

grid_z_magn1 = griddata((x_magn, y_magn), z_magn, (grid_x_magn1, grid_y_magn1), method="cubic")

plt.figure(figsize=(8, 6))
plt.contourf(grid_x_magn1, grid_y_magn1, grid_z_magn1, 30, cmap="viridis")
plt.colorbar(label="Magnetic Anomaly (nT)")
plt.scatter(x_magn, y_magn, c="k", s=10, label="Data points")
plt.legend()
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Interpolated Magnetic Total Field Map")
plt.show()

# Interpolation method (2): Kriging
value_col = "Anomalia_Residual"
df = df_gravimetry.dropna(subset=[value_col, "Este", "Norte"])

x_grav_ne = df["Este"].values
y_grav_ne = df["Norte"].values
z_grav = df[value_col].values

# --- Define the output grid (adjust resolution as needed) ---
nx, ny = 200, 200  # number of cells in X/Y
grid_x_gravm2 = np.linspace(x_grav_ne.min(), x_grav_ne.max(), nx)
grid_y_gravm2 = np.linspace(y_grav_ne.min(), y_grav_ne.max(), ny)

# --- Build the Kriging model ---
# variogram_model: 'linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect'
# You can start with 'spherical' (common in geophysics); try others if needed.
OK = OrdinaryKriging(
    x_grav_ne, y_grav_ne, z_grav,
    variogram_model="spherical",
    nlags=12,                      # number of lag bins for variogram
    verbose=False,
    enable_plotting=False,
    # Optional anisotropy (uncomment/tune if your survey is elongated)
    # anisotropy_scaling=1.5,
    # anisotropy_angle=30.0,       # degrees, counter-clockwise from +x
)

# --- Execute on grid ---
zgrid_gravm2, ssgrid_gravm2 = OK.execute("grid", grid_x_gravm2, grid_y_gravm2)

# --- Plot estimate ---
plt.figure(figsize=(7, 6))
cp = plt.contourf(grid_x_gravm2, grid_y_gravm2, zgrid_gravm2, levels=30, cmap="seismic")
plt.scatter(x_grav_ne, y_grav_ne, s=10, c='k')
plt.colorbar(cp, label=f"{value_col} (units)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title(f"Ordinary Kriging — {value_col}")
plt.show()

# --- Plot kriging variance (uncertainty) ---
plt.figure(figsize=(7, 6))
cp2 = plt.contourf(grid_x_gravm2, grid_y_gravm2, ssgrid_gravm2, levels=30, cmap="seismic")
plt.scatter(x_grav_ne, y_grav_ne, s=10, c='k')
plt.colorbar(cp2, label="Kriging variance")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title(f"Kriging Variance — {value_col}")
plt.show()

value_col = "Anomalia_Magnetica_Campo_Total"
df = df_magnetometry.dropna(subset=[value_col, "Este", "Norte"]).copy()

# --- 1) De-duplicate exact coordinates (average duplicates) ---
df = (df.groupby(["Este", "Norte"], as_index=False)[value_col].mean())

x_magn_ = df["Este"].values
y_magn_ = df["Norte"].values
z_magn = df[value_col].values

x0, y0 = x_magn_.mean(), y_magn_.mean()
x_magn = (x_magn_ - x0) / 1000.0
y_magn = (y_magn_ - y0) / 1000.0


# --- 3) Detrend: fit a plane z ≈ a + b*xk + c*yk ---
G = np.c_[np.ones_like(x_magn), x_magn, y_magn]
coef, *_ = np.linalg.lstsq(G, z_magn, rcond=None)
a, b, c = coef
z_trend = a + b*x_magn + c*y_magn
z_res = z_magn - z_trend

print("z stats:", np.min(z_magn), np.max(z_magn), np.std(z_magn))
print("Residual std:", np.std(z_res))

nx, ny = 250, 250
gridx = np.linspace(x_magn.min(), x_magn.max(), nx)
gridy = np.linspace(y_magn.min(), y_magn.max(), ny)

# --- 5) Variogram parameters (manual) ---
# Good starting guesses:
sill = float(np.var(z_res))
rng = 0.3 * max(x_magn.max()-x_magn.min(), y_magn.max()-y_magn.min())  # ~30% of survey span
nug = 0.05 * sill                                     # small nugget to stabilize

OK1 = OrdinaryKriging(
    x_magn, y_magn, z_res,
    variogram_model="spherical",
    variogram_parameters={"sill": sill, "range": rng, "nugget": nug},
    nlags=12,                      # number of lag bins for variogram
    verbose=False,
    enable_plotting=False,
    # Optional anisotropy (uncomment/tune if your survey is elongated)
    # anisotropy_scaling=1.5,
    # anisotropy_angle=30.0,       # degrees, counter-clockwise from +x
)

# --- Execute on grid ---
zres_grid, ssgrid = OK1.execute("grid", gridx, gridy)

# --- 7) Add the trend back on the grid ---
GX, GY = np.meshgrid(gridx, gridy)         # note: shape (ny, nx)
ztrend_grid = a + b*GX + c*GY
zgrid = zres_grid + ztrend_grid

# --- 8) Back to original meters for plotting axis labels ---
gridx_m = gridx * 1000.0 + x0
gridy_m = gridy * 1000.0 + y0

# --- Plot estimate ---
plt.figure(figsize=(7, 6))
cp = plt.contourf(gridx_m, gridy_m, zgrid, levels=30, cmap="viridis")
plt.scatter(x_magn_, y_magn_, s=10, c='k', alpha=0.6)
plt.colorbar(cp, label=f"{value_col} (units)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title(f"Ordinary Kriging — {value_col}")
plt.show()

# --- Plot kriging variance (uncertainty) ---
plt.figure(figsize=(7, 6))
cp2 = plt.contourf(gridx_m, gridy_m, ssgrid, levels=30, cmap="viridis")
plt.scatter(x_magn_, y_magn_, s=10, c='k')
plt.colorbar(cp2, label="Kriging variance")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title(f"Kriging Variance — {value_col}")
plt.show()

# Full plot (gravimetry)
fig, axs = plt.subplots(figsize=(16, 5), ncols=3)
fig.subplots_adjust(hspace=2)

fig1 = axs[0].scatter(df_gravimetry["Este"], df_gravimetry["Norte"], c=df_gravimetry["Anomalia_Residual"], cmap="seismic")
axs[0].set_title('Gravity Residual Anomaly Map')
axs[0].set_xlabel("East (m)")
axs[0].set_ylabel("North (m)")
fig.colorbar(fig1, ax=axs[0], label="Residual Gravity (mGal)")

fig2 = axs[1].contourf(grid_x_gravm1, grid_y_gravm1, grid_z_gravm1, 30, cmap="seismic")
axs[1].scatter(df_gravimetry["Este"].values, df_gravimetry["Norte"].values, c="k", s=6, label="Data points")
axs[1].set_title('Gravity Anomaly - Bicubic interpolation')
axs[1].set_xlabel("East (m)")
axs[1].set_ylabel("North (m)")
fig.colorbar(fig2, ax=axs[1], label="Residual Gravity (mGal)")

fig3 = axs[2].contourf(grid_x_gravm2, grid_y_gravm2, zgrid_gravm2, 30, cmap="seismic")
axs[2].scatter(df_gravimetry["Este"].values, df_gravimetry["Norte"].values, c="k", s=6, label="Data points")
axs[2].set_title('Gravity Anomaly - Kriging interpolation')
axs[2].set_xlabel("East (m)")
axs[2].set_ylabel("North (m)")
fig.colorbar(fig3, ax=axs[2], label="Residual Gravity (mGal)")

fig.tight_layout()
plt.show()

# Full plot (magnetometry)
fig, axs = plt.subplots(figsize=(16, 5), ncols=3)
fig.subplots_adjust(hspace=2)

fig1 = axs[0].scatter(df_magnetometry["Este"], df_magnetometry["Norte"], c=df_magnetometry["Anomalia_Magnetica_Campo_Total"], cmap="viridis")
axs[0].set_title('Magnetic Total Field Map')
axs[0].set_xlabel("East (m)")
axs[0].set_ylabel("North (m)")
fig.colorbar(fig1, ax=axs[0], label="Magnetic Anomaly (nT)")

fig2 = axs[1].contourf(grid_x_magn1, grid_y_magn1, grid_z_magn1, 30, cmap="viridis")
axs[1].scatter(df_magnetometry["Este"].values, df_magnetometry["Norte"].values, c="k", s=6, label="Data points")
axs[1].set_title('Magnetic Total Field Map - Bicubic interpolation')
axs[1].set_xlabel("East (m)")
axs[1].set_ylabel("North (m)")
fig.colorbar(fig2, ax=axs[1], label="Magnetic Anomaly (nT)")

fig3 = axs[2].contourf(gridx_m, gridy_m, zgrid, 30, cmap="viridis")
axs[2].scatter(x_magn_, y_magn_, c="k", s=6, label="Data points")
axs[2].set_title('Magnetic Total Field Map - Kriging interpolation')
axs[2].set_xlabel("East (m)")
axs[2].set_ylabel("North (m)")
fig.colorbar(fig3, ax=axs[2], label="Magnetic Anomaly (nT)")

fig.tight_layout()
plt.show()

# Full plot (variance)
fig, axs = plt.subplots(figsize=(16, 5), ncols=2)
fig.subplots_adjust(hspace=2)

fig1 = axs[0].contourf(grid_x_gravm2, grid_y_gravm2, ssgrid_gravm2, levels=30, cmap="seismic")
axs[0].set_title('Kriging variance — Gravity Residual Anomaly Map')
axs[0].set_xlabel("East (m)")
axs[0].set_ylabel("North (m)")
fig.colorbar(fig1, ax=axs[0], label="Kriging variance")

fig2 = axs[1].contourf(gridx_m, gridy_m, ssgrid, 30, cmap="seismic")
axs[1].scatter(x_magn_, y_magn_, c="k", s=6, label="Data points")
axs[1].set_title('Kriging variance - Magnetic Total Field Map')
axs[1].set_xlabel("East (m)")
axs[1].set_ylabel("North (m)")
fig.colorbar(fig2, ax=axs[1], label="Kriging variance")

fig.tight_layout()
plt.show()

##
t=0