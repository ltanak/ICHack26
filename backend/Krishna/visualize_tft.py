"""
TFT Fire Spread Prediction Visualization.

Run with: python -m Krishna.visualize_tft
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import sys
sys.path.insert(0, '/dcs/23/u5561033/ICHack26/backend')

from Krishna.tft_predictor import predict_fire_params
from Krishna.simulation import step, create_grid, monte_carlo_simulation, EMPTY, TREE, BURNING, BURNT


def main():
    # Get TFT prediction
    params = predict_fire_params(temperature=50, humidity=5, wind_speed=20, wind_direction=270)

    # Exaggerated fire parameters
    ignition_prob = 1
    wind_strength = 11
    wind_dir = params['wind_dir']
    p_tree = 0.9

    # Load initial grid
    grid, ca_mask = create_grid(p_tree=p_tree)

    # Expand initial fires
    fire_points = np.where(grid == BURNING)
    if len(fire_points[0]) > 0:
        for i in range(len(fire_points[0])):
            y, x = fire_points[0][i], fire_points[1][i]
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and ca_mask[ny, nx] == 1:
                        if grid[ny, nx] == TREE:
                            grid[ny, nx] = BURNING

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15, right=0.85)
    ax.axis('off')
    ax.set_title('Temporal Fusion Transformer Model Fire Spread Prediction', fontsize=14, pad=10)

    # Colormap
    cmap = ListedColormap(['lightgrey', 'green', 'red', 'peru'])
    masked_grid = np.ma.masked_where(ca_mask == 0, grid)
    im = ax.imshow(masked_grid, cmap=cmap, origin='lower', vmin=0, vmax=3)

    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add Monte Carlo checkbox and buttons
    ax_mc = plt.axes([0.86, 0.7, 0.12, 0.06])
    ax_reset = plt.axes([0.86, 0.6, 0.12, 0.05])
    ax_run = plt.axes([0.86, 0.52, 0.12, 0.05])

    mc_check = CheckButtons(ax_mc, ['Monte Carlo'], [False])
    reset_btn = Button(ax_reset, 'Reset')
    run_btn = Button(ax_run, 'Run MC (20)')

    state = {'monte_carlo_mode': False, 'frame_count': 0, 'grid': grid}

    def on_mc_change(label):
        state['monte_carlo_mode'] = not state['monte_carlo_mode']

    def reset(event):
        state['grid'], _ = create_grid(p_tree=p_tree)
        fire_points = np.where(state['grid'] == BURNING)
        if len(fire_points[0]) > 0:
            for i in range(len(fire_points[0])):
                y, x = fire_points[0][i], fire_points[1][i]
                for dy in range(-6, 7):
                    for dx in range(-6, 7):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < state['grid'].shape[0] and 0 <= nx < state['grid'].shape[1] and ca_mask[ny, nx] == 1:
                            if state['grid'][ny, nx] == TREE:
                                state['grid'][ny, nx] = BURNING

        masked_grid = np.ma.masked_where(ca_mask == 0, state['grid'])
        im.set_data(masked_grid)
        im.set_cmap(cmap)
        im.set_clim(0, 3)
        ax.set_title('Temporal Fusion Transformer Model Fire Spread Prediction', fontsize=14)
        info_text.set_text('')
        state['frame_count'] = 0
        fig.canvas.draw_idle()

    def run_monte_carlo(event):
        if not state['monte_carlo_mode']:
            return

        ax.set_title('TFT Monte Carlo - Running 20 simulations...', fontsize=14)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        burn_prob = monte_carlo_simulation(
            n_runs=20,
            p_tree=p_tree,
            ignition_prob=ignition_prob,
            wind_dir=wind_dir,
            wind_strength=wind_strength
        )

        masked_prob = np.ma.masked_where(ca_mask == 0, burn_prob)
        im.set_cmap('hot')
        im.set_clim(0, 1)
        im.set_data(masked_prob)
        ax.set_title('TFT Monte Carlo Analysis - Burn Probability (20 runs)', fontsize=14)

        mean_burn = np.mean(burn_prob[ca_mask == 1])
        max_burn = np.max(burn_prob)
        info_text.set_text(f'Mean burn prob: {mean_burn:.2f}\nMax burn prob: {max_burn:.2f}')
        fig.canvas.draw_idle()

    mc_check.on_clicked(on_mc_change)
    reset_btn.on_clicked(reset)
    run_btn.on_clicked(run_monte_carlo)

    def update(frame):
        if state['monte_carlo_mode']:
            return [im, info_text]

        state['frame_count'] += 1
        if np.any(state['grid'] == BURNING):
            state['grid'] = step(state['grid'], ignition_prob, wind_dir, wind_strength)
            masked_grid = np.ma.masked_where(ca_mask == 0, state['grid'])
            im.set_data(masked_grid)
            burned = np.sum(state['grid'] == BURNT)
            burning = np.sum(state['grid'] == BURNING)
            info_text.set_text(f'Step: {state["frame_count"]}\nBurning: {burning}\nBurned: {burned}')
        else:
            burned = np.sum(state['grid'] == BURNT)
            info_text.set_text(f'Complete\nTotal Burned: {burned}')
        return [im, info_text]

    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show(block=True)


if __name__ == "__main__":
    main()
