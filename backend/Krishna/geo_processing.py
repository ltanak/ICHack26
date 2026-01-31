# import geopandas as gpd
# import fiona 
# import matplotlib.pyplot as plt

# path = "Datasets/Snapshot/2020_Snapshot/20200101AM.gpkg"

# print("Layers:")
# print(fiona.listlayers(path))

# fire = gpd.read_file(path, layer="newfirepix")
# ca = gpd.read_file(
#     "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
# )
# ca = ca[ca["name"] == "California"]

# fig, ax = plt.subplots(figsize=(6, 8))
# ca.boundary.plot(ax=ax, color="black")
# fire.plot(ax=ax, color="red", markersize=20)

# ax.set_title("Fire detections over California")
# plt.show()

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Parameters
N = 200

# Load California boundary
states = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
)
ca = states[states["name"] == "California"]

# Get bounding box
minx, miny, maxx, maxy = ca.total_bounds

# Create grid
xs = np.linspace(minx, maxx, N)
ys = np.linspace(miny, maxy, N)

ca_mask = np.zeros((N, N), dtype=int)

# Rasterise
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        p = Point(x, y)
        if ca.contains(p).any():
            ca_mask[j, i] = 1

# Visual sanity check
plt.imshow(ca_mask, cmap="gray", origin="lower")
plt.title("California mask (1 = inside CA)")
plt.axis("off")
plt.show()
