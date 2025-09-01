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