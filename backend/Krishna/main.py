import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

EMPTY, TREE, BURNING, BURNT = 0, 1, 2, 3
N = 100
ignite_prob = 0.7

def initialise_grid(N, p):
    grid = np.random.choice(
        [EMPTY, TREE],
        size = (N, N),
        p = [1-p, p]
    )
    grid[:, 0][grid[:, 0] == TREE] = BURNING
    return grid

def step(grid, ignite_prob=1):
    N = grid.shape[0]
    new_grid = grid.copy()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] == BURNING:
                new_grid[i, j] = BURNT
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < N:
                        if grid[ni, nj] == TREE:
                            if np.random.rand() < ignite_prob:
                                new_grid[ni, nj] = BURNING
    
    return new_grid

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

ax_p = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_ig = plt.axes([0.2, 0.05, 0.6, 0.03])

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
    valinit = ignite_prob,
    valstep = 0.02
)

def reset(val):
    global grid
    grid = initialise_grid(N, p=p_slider.val)
    im.set_data(grid)

p_slider.on_changed(reset)
ignite_prob_slider.on_changed(reset)

def update(frame):
    global grid
    grid = step(grid, ignite_prob=ignite_prob_slider.val)
    im.set_data(grid)
    return [im]

ani = FuncAnimation(fig, update, interval = 100)
plt.show()