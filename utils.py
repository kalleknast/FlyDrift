import numpy as np

RHO_WATER = 1000.0
VISCOSITY = 0.001002  # Pa*s (Dynamic Viscosity of water at 20C)

# Bead weights in grams, from https://muunntungsten.com/products/muunn-100pcs-3-8-6-4mm-tungsten-slotted-beads-fly-tying-material-multi-color-fly-fishing-tungsten-beads?srsltid=AfmBOoq2T42DLMcD1AeMRvpvw_ab8z3P4i4B4Phn8-wKPeFRP045nJeh
bead_weights = {
    '1.5mm': 0.02,
    '2.0mm': 0.062,
    '2.3mm': 0.09,
    '2.5mm': 0.1,
    '2.8mm': 0.157,
    '3.0mm': 0.182,
    '3.3mm': 0.263,
    '3.5mm': 0.301,
    '3.8mm': 0.401,
    '4.0mm': 0.485,
    '4.6mm': 0.737,
    '5.5mm': 1.308,
    '6.4mm': 2.079
    }

# Tippet diameters in mm, from https://www.durhamflyfishing.co.uk/resources/Stroft%20Tippet%20X%20Conversion%20Chart.htm
tippet_diameters = {
    '9x': 0.06,
    '8x': 0.08,
    '7x': 0.1,
    '6x': 0.12,
    '5x': 0.15,
    '4x': 0.18,
    '3x': 0.2,
    '2x': 0.23,
    '1x': 0.25,
    '0x': 0.28
    }

tippet_densities = {'flourocarbon': 1780.0, 'nylon': 1150.0}

def get_river_velocity(y_positions, river_depth, surface_velocity, river_profile_exp):
    """Vectorized velocity profile calculation."""
    vels = np.zeros_like(y_positions)
    
    # Mask for water and air
    mask_water = (y_positions >= 0) & (y_positions <= river_depth)
    mask_above = y_positions > river_depth
    
    # Power law profile in water
    vels[mask_water] = surface_velocity * (y_positions[mask_water] / river_depth)**river_profile_exp
    # Constant surface speed in air (simplified)
    vels[mask_above] = surface_velocity 
    return vels


def get_fly_drag_coeff(v_rel_mag, diam, bugginess=None):
    """
    Docstring for get_fly_drag_coeff
    
    :param v_rel_mag: Description
    :param diam: Description
    :param bugginess: Description
    """
    if v_rel_mag < 1e-6:
        return 0.0
    
    re = (RHO_WATER * v_rel_mag * diam) / VISCOSITY
    cd = (24.0 / re) * (1 + 0.15 * re**0.687) + 0.42 / (1 + 4.25e4 * re**-1.16)

    if bugginess is None:
        return cd
    else:
        return cd * bugginess      


def calculate_drag_coefficient(Re):
    """
    Calculates drag coefficient for a sphere based on Reynolds number.
    Schiller-Naumann correlation.
    """
    if Re <= 0:
        return 0.0  # Should not happen if speed > 0
    
    if Re < 0.1:
        return 24.0 / Re
    elif Re < 1000:
        return (24.0 / Re) * (1 + 0.15 * Re**0.687)
    else:
        return 0.44


def read_drift(fname):
    npz = np.load(fname)
    drift = {k: v for k, v in npz.items()}
    return drift


def calculate_sink_metrics(drift):
    """
    Computes sink rate metrics for the fly.
    
    Args:
        drift (dict): Dictionary containing drift simulation results.
        
    Returns:
        dict: containing:
            - 'time_to_max_depth': time (s) from start to reach deepest point.
            - 'max_sink_rate': maximum downward vertical velocity (m/s).
            - 'average_sink_rate': (start_depth - min_depth) / time_to_max_depth.
            - 'min_y': y coordinate of deepest point (m from bottom).
    """
    y_fly = drift['y'][0, :]
    t = drift['t']
    vy_fly = drift['vy'][0, :] # Vertical velocity of fly
    
    # Deepest location (minimum y, since y=0 is bottom and y=depth is surface)
    max_depth = find_depth_elbow(drift)
    min_y = max_depth['y']
    time_to_max_depth = max_depth['time']
    
    # Sink rate (positive for downward)
    sink_rates = -vy_fly
    max_sink_rate = np.max(sink_rates)
    
    start_y = y_fly[0]
    average_sink_rate = (start_y - min_y) / time_to_max_depth if time_to_max_depth > 0 else 0.0
    
    return {
        'time_to_max_depth': float(time_to_max_depth),
        'max_sink_rate': float(max_sink_rate),
        'average_sink_rate': float(average_sink_rate),
        'min_y': float(min_y), # Distance from bottom
        'max_depth': float(drift.get('river_depth', 1.5) - min_y) # Distance from surface
    }


def calculate_relative_fly_speed(drift):
    """
    Calculates the fly's speed along x relative to the surrounding water.
    
    Args:
        drift (dict): Drift simulation results.
        
    Returns:
        dict: {
            'relative_speed': array of (v_fly / v_water),
            'mean_relative_speed': float,
            'median_relative_speed': float,
            't': array of time
        }
    """
    full_depth = find_depth_elbow(drift)
    vx_fly = drift['vx'][0, full_depth['index']:]
    vx_water = drift['water_vx'][0, full_depth['index']:]
    v_rel = vx_fly / vx_water
    
    return {
        'relative_speed': v_rel,
        'mean_relative_speed': float(np.nanmean(v_rel)),
        'median_relative_speed': float(np.nanmedian(v_rel)),
        't': drift['t']
    }


def find_depth_elbow(drift):
    """
    Finds the 'elbow' point/time where the fly reaches approximately full depth 
    and the sink rate slows down drastically. Uses the Kneedle algorithm (max distance from line).
    
    Args:
        drift (dict): Drift simulation results.
        
    Returns:
        dict: {
            'time': float,
            'index': int,
            'y': float
        }
    """
    y = drift['y'][0, :]
    t = drift['t']
    
    # Considering segment from start to point of minimum depth (deepest point)
    idx_min = np.argmin(y)
    
    if idx_min == 0:
        return {'time': float(t[0]), 'index': 0, 'y': float(y[0])}
        
    t_seg = t[:idx_min+1]
    y_seg = y[:idx_min+1]
    
    # Normalize to unit square [0, 1] to handle different scales of time and depth
    # t moves 0 -> 1
    t_min, t_max = t_seg[0], t_seg[-1]
    denom_t = t_max - t_min
    t_norm = (t_seg - t_min) / denom_t if denom_t > 0 else np.zeros_like(t_seg)
    
    # y (depth) moves 1 -> 0 (since it decreases)
    # y_norm = (y - min) / (max - min). 
    # At start (max y), y_norm=1. At min depth (min y), y_norm=0.
    y_min, y_max = np.min(y_seg), np.max(y_seg)
    denom_y = y_max - y_min
    y_norm = (y_seg - y_min) / denom_y if denom_y > 0 else np.zeros_like(y_seg)
    
    # We look for the point with max distance from the line connecting (0, 1) and (1, 0).
    # Line eq: x + y - 1 = 0
    # Dist = |x + y - 1| / sqrt(2)
    # We want to maximize |t_norm + y_norm - 1|
    
    distances = np.abs(t_norm + y_norm - 1)
    elbow_local_idx = np.argmax(distances)
    
    return {
        'time': float(t[elbow_local_idx]),
        'index': int(elbow_local_idx),
        'y': float(y[elbow_local_idx])
    }