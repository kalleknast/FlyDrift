import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import get_river_velocity, calculate_sink_metrics, calculate_relative_fly_speed, read_drift
from utils import find_depth_elbow

plt.ion()


def plot_drifts(drifts, xlabel, title="Drift", labels=None):
    fig = plt.figure(figsize=(20, 18))
    
    # Grid layout: Main plot is 2 units high, subplots are 1 unit high
    # Total rows = 1 (main) + 4 (subs) = 5
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1])
    
    # Main Plot (Distance vs Depth)
    ax_main = fig.add_subplot(gs[0, 0])
    
    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']
    line_styles = ['-', ':', '-.', '--']
    for i, drift in enumerate(drifts):
        if labels is not None:
            label = labels[i]
        else:
            label = f"rod tip: {drift['rod_tip_speed_factor']}"
        
        ls = line_styles[i // len(cmaps)]
        plot_drift(drift, ax_main, cmap=cmaps[i % len(cmaps)], line_style=ls, label=label)
        
    ax_main.legend(fontsize=14)
    ax_main.set_title(title, fontsize=18)
    
    # --- Metric Calculations ---
    avg_sink_rates = []
    max_depths = []
    mean_rel_speeds = []
    times_to_elbow = []
    
    for drift in drifts:
        m = calculate_sink_metrics(drift)
        avg_sink_rates.append(m['average_sink_rate'])
        max_depths.append(m['max_depth'])
        
        r = calculate_relative_fly_speed(drift)
        mean_rel_speeds.append(r['mean_relative_speed'])
        
        elbow = find_depth_elbow(drift)
        times_to_elbow.append(elbow['time'])
        
    # --- Subplots ---
    if labels is None:
        plot_labels = [str(i) for i in range(len(drifts))]
    else:
        plot_labels = labels
        
    x_idxs = np.arange(len(drifts))

    marker_colors = [colormaps[cmaps[i % len(cmaps)]](0.9) for i in range(len(drifts))]

    # Subplot 1: Average Sink Rate
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x_idxs, avg_sink_rates, ':', color='k', linewidth=2)
    ax1.scatter(x_idxs, avg_sink_rates, color=marker_colors, s=50, zorder=3, edgecolors=marker_colors)
    ax1.set_ylabel("Avg Sink Rate (m/s)", fontsize=12)
    ax1.grid(True)
    ax1.set_xticklabels([])  # Hide x labels for top subplots
    
    # Subplot 2: Max Depth (min_y)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax2.plot(x_idxs, max_depths, ':', color='k', linewidth=2)
    ax2.scatter(x_idxs, max_depths, color=marker_colors, s=50, zorder=3, edgecolors=marker_colors)
    ax2.set_ylabel("Max Depth (m)", fontsize=12)
    ax2.grid(True)
    ax2.set_xticklabels([]) 

    # Subplot 3: Mean Relative Fly Speed
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax3.plot(x_idxs, mean_rel_speeds, ':', color='k', linewidth=2)
    ax3.scatter(x_idxs, mean_rel_speeds, color=marker_colors, s=50, zorder=3, edgecolors=marker_colors)
    ax3.set_ylabel("Mean Rel. Speed (m/s)", fontsize=12)
    ax3.grid(True)
    ax3.set_xticklabels([])

    # Subplot 4: Time to Elbow Depth
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax4.plot(x_idxs, times_to_elbow, ':', color='k', linewidth=2)
    ax4.scatter(x_idxs, times_to_elbow, color=marker_colors, s=50, zorder=3, edgecolors=marker_colors)
    ax4.set_ylabel("Time to Elbow (s)", fontsize=12)
    ax4.grid(True)
    
    # Set shared x ticks
    ax4.set_xticks(x_idxs)
    ax4.set_xticklabels(plot_labels, rotation=0, ha='center', fontsize=12)
    ax4.set_xlabel(xlabel, fontsize=14)

    plt.tight_layout()
    return ax_main


def plot_drift(drift, ax=None, cmap='Blues', line_style='-', label=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))

    # Add flow profile subplot if not already present
    if not getattr(ax, '_has_flow_profile', False):
        divider = make_axes_locatable(ax)
        ax_profile = divider.append_axes("left", size="20%", pad=0.0)
        
        rd = drift.get('river_depth', 1.5)
        # Handle potential array wrapping of scalar
        if np.ndim(rd) > 0: rd = float(rd.flat[0])
            
        sv = drift.get('surface_velocity', 1.0)
        if np.ndim(sv) > 0: sv = float(sv.flat[0])
        
        # Profile: Simulation y is 0 (bottom) to rd (surface)
        y_sim = np.linspace(0, rd, 100)
        v_vals = get_river_velocity(y_sim, rd, sv, drift['river_profile_exp'])
        
        # Plot coordinates: 0 is surface, rd is bottom (positive depth)
        y_plot = rd - y_sim
        
        ax_profile.plot(v_vals, y_plot, '-', color='blue')
        ax_profile.fill_betweenx(y_plot, 0, v_vals, color='lightblue', alpha=0.5)
        ax_profile.axhline(0, color='blue', linestyle='-', linewidth=1)
        ax_profile.set_ylim(rd, -0.2)
        ax_profile.set_xlim(0, sv)
        ax_profile.set_xlabel("Velocity (m/s)", fontsize=14)
        ax_profile.set_ylabel("Depth (m)", fontsize=14)
        ax_profile.set_title("Flow Profile", fontsize=18)
        ax_profile.grid(True)
        
        # Remove y-labels from main plot since they are now on the profile
        ax.set_yticklabels([])
        ax.set_ylabel("")
        
        ax._has_flow_profile = True

    if label is None:
        label = f'tippet dia: {drift["tippet_diameter"]}, fly size: {drift["fly_mass"]}'

    x = drift['x'][::-1, :]
    y = drift['river_depth'] - drift['y'][::-1, :]
    t = drift['t']

    cmap = colormaps[cmap]

    ax.plot([drift['x'].min(), drift['x'].max()], [0, 0], 'k:', linewidth=1)

    # Plot line states
    for i in range(0, len(t), 5): # Plot every 5th frame
        # Color fades from light blue (start) to dark blue (end)
        color = cmap(0.3 + 0.7 * (i / len(t)))
        ax.plot(x[:, i], y[:, i], color=color, linestyle=line_style)
        ax.plot(x[-1, i], y[-1, i], 'o', color=color, markersize=4) # The Fly
    ax.plot(x[:, i], y[:, i], color=color, linestyle=line_style, label=label)
    
    # if label is not None:
    #     ax.legend()
    ax.set_title(f"{label}", fontsize=18)
    ax.set_xlabel("Distance (m)", fontsize=14)
    ax.set_ylabel("Depth (m)", fontsize=14)
    ax.set_ylim(drift['river_depth'], -0.2)
    ax.set_xlim(drift['x'].min(), drift['x'].max())
    ax.grid(True)
    ax.legend(fontsize=14)

    return ax


def plot_sink_rates(drifts, metric='max_sink_rate', labels=None, title=None, ax=None):
    """
    Plots a comparison of sink rate metrics for multiple drifts.
    
    Args:
        drifts (list): List of drift dictionaries.
        metric (str): Metric to plot ('max_sink_rate', 'average_sink_rate', 'time_to_max_depth', 'min_y').
        labels (list): Labels for each drift.
        title (str): Plot title.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    metrics = []
    for drift in drifts:
        m = calculate_sink_metrics(drift)
        metrics.append(m[metric])
        
    if labels is None:
        labels = [f"Drift {i+1}" for i in range(len(drifts))]
        
    x = np.arange(len(metrics))
    ax.bar(x, metrics, align='center', alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ylabel_map = {
        'max_sink_rate': 'Max Sink Rate (m/s)',
        'average_sink_rate': 'Avg Sink Rate (m/s)',
        'time_to_max_depth': 'Time to Max Depth (s)',
        'min_y': 'Deepest Point (m above bottom)',
        'max_depth': 'Max Depth (m)'
    }
    
    ax.set_ylabel(ylabel_map.get(metric, metric))
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Comparison of {ylabel_map.get(metric, metric)}")
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return ax


    return ax


def plot_relative_fly_speed(drifts, labels=None, title=None, ax=None):
    """
    Plots the fly's relative speed over time for multiple drifts.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    if labels is None:
        labels = [f"Drift {i+1}" for i in range(len(drifts))]
        
    means = np.zeros(len(drifts))
    medians = np.zeros(len(drifts))
    for i, drift in enumerate(drifts):
        res = calculate_relative_fly_speed(drift)
        means[i] = res['mean_relative_speed']
        medians[i] = res['median_relative_speed']
        
    # Simple line plot
    ax.plot(means, 's:', label='means') # , label=f"{labels[i]} (Mean: {means[i]:.3f} m/s)")
    ax.plot(medians, 'o:', label='medians') # , label=f"{labels[i]} (Median: {median_v:.3f} m/s)")
        
    ax.set_xlabel("Drift")
    ax.set_xticks(np.arange(len(drifts)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Fly relative speed to water (m/s)")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8) # Zero relative speed
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Fly Relative Speed vs Time")
        
    ax.legend()
    ax.grid(True)
    return ax


def make_movie(drifts, filename='drift_movie.gif', fps=10, line_colors=None, labels=None, title=None):
    """
    Creates an animation of one or more drifts.

    Args:
        drifts (list or dict): Single drift dict or list of drift dicts.
        filename (str): Output filename (e.g., 'drift.mp4' or 'drift.gif').
        fps (int): Frames per second.
        line_colors (list): List of colors for the lines.
        labels (list): List of labels for the legend.
    """
    # Normalize inputs to list
    if isinstance(drifts, dict):
        drifts = [drifts]
    
    # Defaults
    if line_colors is None:
        line_colors = ['tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple']
        line_styles = ['-', ':', '-.', '--']
    if labels is None:
        labels = [f"Drift {i+1}" for i in range(len(drifts))]

    if title is None:
        title = "Drift Animation"
        
    # Ensure color list is long enough
    while len(line_colors) < len(drifts):
        line_colors += line_colors

    # Use environment parameters from the first drift
    drift0 = drifts[0]
    river_depth = drift0.get('river_depth', 1.5)
    if np.ndim(river_depth) > 0: river_depth = float(river_depth.flat[0])
    
    surface_velocity = drift0.get('surface_velocity', 1.4)
    if np.ndim(surface_velocity) > 0: surface_velocity = float(surface_velocity.flat[0])

    river_profile_exp = drift0.get('river_profile_exp', 0.166)
    if np.ndim(river_profile_exp) > 0: river_profile_exp = float(river_profile_exp.flat[0])
    
    t = drift0['t']
    num_steps = len(t)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    
    # Determine global plot limits across all drifts
    max_depth_glob = river_depth
    min_x_glob = float('inf')
    max_x_glob = float('-inf')

    for d in drifts:
        # Check depth
        rd = d.get('river_depth', 1.5)
        if np.ndim(rd) > 0: rd = float(rd.flat[0])
        max_depth_glob = max(max_depth_glob, rd)
        
        # Check x range
        min_x_glob = min(min_x_glob, np.min(d['x']))
        max_x_glob = max(max_x_glob, np.max(d['x']))

    ax.set_ylim(max_depth_glob * 1.1, -0.2)
    
    pad = 1.0
    xlim_min, xlim_max = min_x_glob - pad, max_x_glob + pad
    ax.set_xlim(xlim_min, xlim_max)
    
    # Static elements
    # ax.axhline(0, color='brown', linestyle='--', linewidth=2, label="River Bed")
    # ax.axhline(river_depth, color='blue', linestyle='-', linewidth=1, label="Surface")
    ax.axhspan(0, river_depth, color='lightcyan', alpha=0.3)
    ax.grid(True)
    
    # Initialize Line objects for each drift
    lines = []
    fly_dots = []
    
    for i, _ in enumerate(drifts):
        ls = line_styles[i // len(line_styles)]
        ln, = ax.plot([], [], color=line_colors[i], label=labels[i], linestyle=ls)
        fd, = ax.plot([], [], 'o', color=line_colors[i], markersize=4) # Fly is black
        lines.append(ln)
        fly_dots.append(fd)
        
    ax.legend(loc='upper right')
    
    # Water particles setup (Regular Grid)
    nx = 20  # Number of particles along X
    ny = 10  # Number of particles along Y (depth)
    x_grid = np.linspace(xlim_min, xlim_max, nx)
    y_grid_sim = np.linspace(0, river_depth, ny)
    particle_x, particle_y_sim = np.meshgrid(x_grid, y_grid_sim)
    particle_x = particle_x.flatten()
    particle_y_sim = particle_y_sim.flatten()
    num_particles = len(particle_x)
    
    # Use LineCollection for water segments instead of dots
    water_segments = LineCollection([], colors='cornflowerblue', alpha=0.8, linewidths=1.5)
    ax.add_collection(water_segments)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Pre-calculate durations between frames
    dt_frames = np.diff(t)
    dt_frames = np.append(dt_frames, dt_frames[-1])
    
    def init():
        for ln in lines: ln.set_data([], [])
        for fd in fly_dots: fd.set_data([], [])
        water_segments.set_segments([])
        time_text.set_text('')
        return (*lines, *fly_dots, water_segments, time_text)

    def update(frame):
        # Update each drift
        for i, d in enumerate(drifts):
            # Safe check if drifts have different lengths (not expected but good practice)
            if frame < d['x'].shape[1]:
                xs = d['x'][:, frame]
                rd = d.get('river_depth', 1.5)
                if np.ndim(rd) > 0: rd = float(rd.flat[0])
                ys = rd - d['y'][:, frame]
                lines[i].set_data(xs, ys)
                fly_dots[i].set_data([xs[0]], [ys[0]])

        # Update water particles
        dt = dt_frames[frame] if frame < len(dt_frames) else 0.1
        v_particles_ground = get_river_velocity(particle_y_sim, river_depth, surface_velocity, river_profile_exp)
        
        particle_x[:] += v_particles_ground * dt
        
        # Wrap particles
        width = xlim_max - xlim_min
        for k in range(num_particles):
            if particle_x[k] < xlim_min:
                particle_x[k] += width
            elif particle_x[k] > xlim_max:
                particle_x[k] -= width
        
        # Create line segments where length is proportional to velocity
        # Scale factor controls the visual length
        scale = 0.2
        half_lens = (v_particles_ground * scale) / 2.0
        
        # Construct segments array: (num_particles, 2, 2)
        # Point 1: (x - len/2, y)
        # Point 2: (x + len/2, y)
        # Plot coordinates (Depth)
        particle_y_plot = river_depth - particle_y_sim

        segments = np.zeros((num_particles, 2, 2))
        segments[:, 0, 0] = particle_x - half_lens
        segments[:, 0, 1] = particle_y_plot
        segments[:, 1, 0] = particle_x + half_lens
        segments[:, 1, 1] = particle_y_plot
        
        water_segments.set_segments(segments)
        time_text.set_text(f"Time: {t[frame]:.2f}s")
        
        return (*lines, *fly_dots, water_segments, time_text)
    
    anim = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)
    
    print(f"Saving animation to {filename}...")
    if filename.endswith('.gif'):
        anim.save(filename, writer='pillow', fps=fps)
    else:
        try:
            anim.save(filename, writer='ffmpeg', fps=fps)
        except Exception as e:
            print(f"Error saving video with ffmpeg: {e}. Trying .gif instead.")
            anim.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps)
            
    print("Done.")


def plot_heatmap(
    x_values, 
    y_values, 
    z_name, 
    labels, 
    path='data', 
    tippet_diameter=None, 
    fly_mass=None, 
    fly_diameter=None, 
    rod_tip_speed_factor=None,
    bugginess=None
):
    """
    Plots a heatmap of a dependent variable (z) against two independent variables (x, y).
    
    Args:
        x_values (list): List of values for the x-axis.
        y_values (list): List of values for the y-axis.
        z_name (str): Name of the dependent variable (e.g., 'average sink rate').
        labels (dict): Labels for the plot (keys: 'xlabel', 'ylabel', 'zlabel').
        path (str): Path to the simulation data directory.
        tippet_diameter (str, optional): Fixed line diameter/tippet (e.g., '6x'). 
                                       Required if not the x or y variable.
        fly_mass (str, optional): Fixed fly mass/bead size (e.g., '2.5mm').
                                  Required if not the x or y variable.
        fly_diameter (float, optional): Fixed fly diameter (e.g., 0.003).
                                        Required if not the x or y variable.
        rod_tip_speed_factor (float, optional): Fixed rod tip speed factor.
                                                Required if not the x or y variable.
    """
    
    # Helper to map label names to parameter keys
    def get_param_key(label_text):
        lt = label_text.lower()
        if 'rod' in lt or 'speed' in lt: return 'rod_tip_speed_factor'
        if 'weight' in lt or 'mass' in lt or 'bead' in lt: return 'fly_mass'
        if 'fly diameter' in lt: return 'fly_diameter'
        if 'line' in lt or 'tippet' in lt: return 'tippet_diameter'
        if 'bugginess' in lt: return 'bugginess'
        return None

    # Identify x and y parameters
    x_param = get_param_key(labels.get('xlabel', ''))
    y_param = get_param_key(labels.get('ylabel', ''))
    
    if not x_param or not y_param:
        raise ValueError("Could not infer parameter types from xlabel/ylabel. Please ensure labels contain descriptive names (e.g. 'rod speed', 'fly weight', 'line diameter').")

    # Mapping from parameter name to filename abbreviation
    param_abbr = {
        'rod_tip_speed_factor': 'rs',
        'fly_mass': 'fm',
        'fly_diameter': 'fd',
        'tippet_diameter': 'ld',
        'bugginess': 'bs'
    }

    # Prepare matrix
    Z = np.zeros((len(y_values), len(x_values)))
    Z[:] = np.nan
    
    # Fixed parameters dict
    params = {
        'rod_tip_speed_factor': rod_tip_speed_factor,
        'fly_mass': fly_mass,
        'fly_diameter': fly_diameter,
        'tippet_diameter': tippet_diameter,
        'bugginess': bugginess
    }
    
    # Check for missing fixed parameters
    required_params = ['rod_tip_speed_factor', 'fly_mass', 'fly_diameter', 'tippet_diameter', 'bugginess']
    for rp in required_params:
        if rp != x_param and rp != y_param and params[rp] is None:
             # Try to find a default if only one exists in the directory?
             # For now, strict requirement as per prompt "add necessary arguments"
             # But maybe we can list files and see if there is a unique value.
             print(f"Warning: {rp} was not provided and is not an axis variable. Ambiguity may occur.")
    
    # Iterate and fill matrix
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            
            # Set current values
            current_params = params.copy()
            current_params[x_param] = x_val
            current_params[y_param] = y_val
            
            # Construct filename
            # Format: drift_ld{}_fm{}_fd{}_rs{}.npz
            # Note: Need to match the string formatting in simulations.py exactly
            
            ld = current_params['tippet_diameter']
            fm = current_params['fly_mass']
            fd = current_params['fly_diameter']
            rs = current_params['rod_tip_speed_factor']
            bs = current_params['bugginess']
            
            if None in [ld, fm, fd, rs, bs]:
                # Skip if we don't have all params (or try to wildcard search?)
                # To follow strict prompt instructions about ambiguity, we really should have these.
                # However, if strictness prevents ANY plotting, that's bad.
                # Let's try to wildcard search if we have partials.
                continue

            fname = f"drift_ld{ld}_fm{fm}_fd{fd}_rs{rs}_bs{bs}.npz"
            full_path = os.path.join(path, fname)
            
            if os.path.exists(full_path):
                drift = read_drift(full_path)
                
                # Extract Z variable
                # Assuming z_name maps to a metric
                # First, check direct keys
                # Then calculate metrics
    
                val = np.nan
                
                # Normalize z_name for comparison
                z_key = z_name.lower().replace(' ', '_')
                
                metrics = calculate_sink_metrics(drift)
                rel_speed = calculate_relative_fly_speed(drift)
                
                if z_key in metrics:
                    val = metrics[z_key]
                elif z_key in rel_speed: # e.g. mean_relative_speed
                    val = rel_speed[z_key]
                # Handle 'average sink rate' explicitly if needed (it matches metrics key)
                
                Z[i, j] = val
            # else:
            #     print(f"File not found: {fname}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use pcolormesh or imshow
    # x/y values might not be evenly spaced, so pcolormesh is better, but requires meshgrid
    
    # If values are strings (like '2.0mm'), we need equidistant indices
    use_indices = False
    if isinstance(x_values[0], str) or isinstance(y_values[0], str):
        use_indices = True
        
    if use_indices:
        im = ax.imshow(Z, cmap='viridis', origin='lower', aspect='auto')
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
    else:
        # Check if they are numeric
        try:
            X, Y = np.meshgrid(x_values, y_values)
            im = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        except:
             # Fallback to indices if meshgrid fails (e.g. mixed types)
            im = ax.imshow(Z, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks(np.arange(len(x_values)))
            ax.set_yticks(np.arange(len(y_values)))
            ax.set_xticklabels(x_values)
            ax.set_yticklabels(y_values)
            

    cbar = plt.colorbar(im, ax=ax)
    if 'zlabel' in labels:
        cbar.set_label(labels['zlabel'], rotation=270, labelpad=15)
        
    ax.set_xlabel(labels.get('xlabel', 'X'), fontsize=14)
    ax.set_ylabel(labels.get('ylabel', 'Y'), fontsize=14)
    ax.set_title( f"{labels.get('zlabel', z_name)}", fontsize=18)
    
    return fig, ax
