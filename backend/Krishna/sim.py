import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.colors import ListedColormap


EMPTY, TREE, BURNING, BURNT = 0, 1, 2, 3
N = 100
ANIMATION_INTERVAL = 100
MONTE_CARLO_RUNS = 50

base_ignite_prob = 0.7

WIND_DIRS = {
    "None": (0, 0),
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}


def initialise_grid(N, p):
    grid = np.random.choice(
        [EMPTY, TREE],
        size = (N, N),
        p = [1-p, p]
    )
    grid[:, 0][grid[:, 0] == TREE] = BURNING
    return grid

def get_wind_multiplier(direction, wind_dir, wind_str):
    if direction == wind_dir:
        return wind_str * 2
    elif direction == (-wind_dir[0], -wind_dir[1]):
        return 1 / (wind_str * 2)
    return 1

def step(grid, base_ignite_prob, wind_dir, wind_str):
    N = grid.shape[0]
    new_grid = grid.copy()

    burning_cells = np.argwhere(grid == BURNING)

    for i, j in burning_cells:
        new_grid[i, j] = BURNT
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if grid[ni, nj] == TREE:
                    multiplier = get_wind_multiplier(direction=(di, dj), wind_dir=wind_dir, wind_str=wind_str)
                    if np.random.rand() < min(1, base_ignite_prob * multiplier):
                        new_grid[ni, nj] = BURNING
    return new_grid

def monte_carlo(
        runs,
        grid_shape, 
        density,
        base_ignite_prob,
        wind_dir,
        wind_str
):
    burn_counts = np.zeros((grid_shape, grid_shape))

    for _ in range(runs):
        grid = initialise_grid(N, density)

        while np.any(grid == BURNING):
            if not np.any(grid == TREE):
                break
            grid = step(grid, base_ignite_prob, wind_dir, wind_str)

        burn_counts += (grid == BURNT)

    return burn_counts / runs

cmap = ListedColormap([
    "lightgrey",
    "green",
    "red",
    "peru"
])

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.axis("off")

p0 = 0.9

grid = initialise_grid(N, p0)

im = ax.imshow(grid, cmap=cmap, vmin = 0, vmax = 3)

ax_mode = plt.axes([0.02, 0.7, 0.12, 0.15])
ax_p = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_ig = plt.axes([0.2, 0.06, 0.6, 0.03])
ax_wind = plt.axes([0.2, 0.02, 0.6, 0.03])
ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2])
ax_reset = plt.axes([0.85, 0.85, 0.1, 0.05])

p_slider = Slider(
    ax = ax_p,
    label='Vegetation density',
    valmin=0.1,
    valmax=0.9,
    valinit=p0,
    valstep=0.02
)

ignite_prob_slider = Slider(
    ax = ax_ig,
    label = 'Dryness',
    valmin = 0.1,
    valmax = 0.9,
    valinit = base_ignite_prob,
    valstep = 0.02
)

wind_slider = Slider(
    ax=ax_wind,
    label="Wind strength",
    valmin=1.0,
    valmax=9.0,
    valinit=1.0,
    valstep=0.1
)

radio = RadioButtons(ax_radio, ("None", "Up", "Down", "Left", "Right"))
mode_radio = RadioButtons(ax_mode, ("Single Run", "Monte Carlo"))
mode = "Single Run"


def set_mode(label):
    global mode
    mode = label
    reset(None)

mode_radio.on_clicked(set_mode)

wind_dir = (0, 0)

def set_wind(label):
    global wind_dir
    wind_dir = WIND_DIRS[label]

radio.on_clicked(set_wind)

def reset(val):
    global grid
    if mode == "Single Run":
        grid = initialise_grid(N, p=p_slider.val)
        im.set_cmap(cmap)
        im.set_clim(0, 3)
        im.set_data(grid)
    elif mode == "Monte Carlo":
        burn_prob = monte_carlo(
            runs = MONTE_CARLO_RUNS,
            grid_shape=N,
            density=p_slider.val,
            base_ignite_prob=ignite_prob_slider.val,
            wind_dir=wind_dir,
            wind_str=wind_slider.val
        )

        im.set_cmap("hot")
        im.set_clim(0, 1)
        im.set_data(burn_prob)

p_slider.on_changed(reset)
ignite_prob_slider.on_changed(reset)
wind_slider.on_changed(reset)
radio.on_clicked(reset)
reset_btn = Button(ax_reset, 'Reset')
reset_btn.on_clicked(reset)

def update(frame):
    global grid

    if mode == "Single Run":
        if not np.any(grid == BURNING):
            return [im]
        grid = step(grid, base_ignite_prob = p_slider.val, wind_dir = wind_dir, wind_str = wind_slider.val)
        im.set_data(grid)
    
    return [im]

ani = FuncAnimation(fig, update, interval = ANIMATION_INTERVAL)
plt.show()
