import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import utils as ut

# --- Physics Constants ---
G = 9.81
RHO_WATER = 1000.0
VISCOSITY = 0.001002        # Pa*s (Dynamic Viscosity of water at 20C)
# C_DAMPING = 0.5           # Internal damping to kill high-freq vibrations
# K_GROUND = 2000.0         # Spring constant for river bed interaction (N/m)
# C_GROUND = 20.0           # Damping constant for river bed interaction
TARGET_STRAIN = 0.01        # 1% max stretch
DAMPING_RATIO = 0.9
TIPPET_DENSITY = 1780.0     # flourocarbon density kg/m^3
# --- Derived Drag Parameters ---
CD_TIPPET_NORM = 1.2
CD_TIPPET_TAN = 0.01


def simulate(
    fly_mass=0.0003,
    fly_diameter=0.003,       # Core diameter (Buoyancy)
    bugginess=2.0,            # fly bushiness 1-4
    tippet_diameter=0.0002,
    tippet_length=1.5,
    rod_tip_speed_factor=0.8,
    river_depth=1.5,
    surface_velocity=1.0,
    river_profile_exp=0.166,
    time_span=15.0,
    segments=20
):

    if type(tippet_diameter) is str:
        tippet_diameter = ut.tippet_diameters[tippet_diameter] / 1000.0
    if type(fly_mass) is str:
        fly_mass = ut.bead_weights[fly_mass] / 1000.0
    
    # A. Calculate Node Mass
    # Line
    vol_tippet_total = np.pi * (tippet_diameter/2)**2 * tippet_length
    mass_tippet_total = vol_tippet_total * TIPPET_DENSITY
    mass_node_tippet = mass_tippet_total / segments
    # Fly volumne - matters for buoyancy calculations
    # but ignore this since the vol_fly isn't air filled.
    # vol_fly = (4/3) * np.pi * (fly_diameter/2)**3
    
    # B. Stiffness (K) to enforce < 0.5% stretch
    # We use a fictitious Young's Modulus if needed to ensure rigidity.
    # Target: Under a typical load of ~0.2N (heavy drag), strain < 0.005
    # F = k * dx  ->  F = k * (strain * L_seg)
    # k = F / (strain * L_seg)
    dm_tippet = tippet_length / segments
    estimated_max_load = 0.05  # Newtons (Conservative upper bound for drag+gravity)
    
    k_spring_min = estimated_max_load / (TARGET_STRAIN * dm_tippet)
    
    # We set K to the higher of physical nylon OR this enforcement value.
    # Young's Modulus for stiff braid/spectra is ~100 GPa.
    # Let's just use the enforced stiffness to guarantee user requirement.
    k_spring = max(k_spring_min, 100.0)  # Ensure it's at least 100 N/m
    
    # C. Critical Damping (The "Anti-Bounce" Factor)
    # c_crit = 2 * sqrt(k * m)
    # We damp 90% of critical to allow slight motion but kill oscillation.
    c_damping = DAMPING_RATIO * 2 * np.sqrt(k_spring * mass_node_tippet)

    # --- Vectorized Physics ---
    def system_derivatives_vec(t, state):
        n_nodes = segments + 1
        pos = state[:2*n_nodes].reshape(n_nodes, 2)
        vel = state[2*n_nodes:].reshape(n_nodes, 2)
        
        forces = np.zeros_like(pos)
        
        # 1. Environmental
        v_water_x = ut.get_river_velocity(pos[:, 1],
                                          river_depth,
                                          surface_velocity,
                                          river_profile_exp)
        v_water = np.column_stack((v_water_x, np.zeros(n_nodes)))
        v_rel = v_water - vel
        v_rel_mag = np.linalg.norm(v_rel, axis=1)
        is_submerged = pos[:, 1] <= river_depth
        
        # Gravity
        masses = np.full(n_nodes, mass_node_tippet)
        masses[0] = fly_mass
        forces[:, 1] -= masses * G
        
        # Buoyancy
        buoy_forces = np.zeros(n_nodes)
        buoy_forces[1:] = (mass_node_tippet / TIPPET_DENSITY) * RHO_WATER * G
        # Fly buoyancy since the vol_fly isn't air filled.
        # buoy_forces[0] = vol_fly * RHO_WATER * G
        forces[is_submerged, 1] += buoy_forces[is_submerged]
        
        # Line Drag
        tangents = pos[1:] - pos[:-1]
        tan_lens = np.linalg.norm(tangents, axis=1)
        valid_tan = tan_lens > 1e-9
        tangents[valid_tan] /= tan_lens[valid_tan, None]
        
        v_rel_tippet = v_rel[1:]
        v_tan_mag = np.sum(v_rel_tippet * tangents, axis=1)
        v_tan_vec = v_tan_mag[:, None] * tangents
        v_norm_vec = v_rel_tippet - v_tan_vec
        v_norm_mag = np.linalg.norm(v_norm_vec, axis=1)
        
        area_proj = tippet_diameter * dm_tippet
        f_norm = (0.5 * RHO_WATER * area_proj * CD_TIPPET_NORM) * v_norm_mag[:, None] * v_norm_vec
        
        area_surf = np.pi * tippet_diameter * dm_tippet
        f_tan = (0.5 * RHO_WATER * area_surf * CD_TIPPET_TAN * np.abs(v_tan_mag)[:, None]) * (tangents * np.sign(v_tan_mag)[:, None])
        
        tippet_submerged = is_submerged[1:]
        forces[1:][tippet_submerged] += (f_norm + f_tan)[tippet_submerged]
        
        # Fly Drag
        if is_submerged[0]:
            speed_fly = v_rel_mag[0]
            if speed_fly > 0:
                cd_fly = ut.get_fly_drag_coeff(speed_fly, fly_diameter, bugginess)
                area_fly = np.pi * (fly_diameter/2)**2
                f_fly_mag = 0.5 * RHO_WATER * area_fly * cd_fly * speed_fly**2
                forces[0] += f_fly_mag * (v_rel[0] / speed_fly)
        
        # 2. Internal Tension (Stiff Spring + Critical Damping)
        deltas = pos[1:] - pos[:-1]
        dists = np.linalg.norm(deltas, axis=1)
        dirs = np.zeros_like(deltas)
        mask_dist = dists > 0
        dirs[mask_dist] = deltas[mask_dist] / dists[mask_dist, None]
        
        # Hooke's Law (Linear Spring - allows compression to avoid Singularity)
        stretch = dists - dm_tippet
        k_force = stretch * k_spring
        
        # Damping (Viscous internal friction)
        # v_diff is velocity of node i+1 relative to i
        v_diff = vel[1:] - vel[:-1]
        v_proj = np.sum(v_diff * dirs, axis=1)
        # Damping resists expansion velocity
        damp_force = c_damping * v_proj
        
        total_tension = k_force + damp_force
        tension_vec = total_tension[:, None] * dirs
        
        forces[:-1] += tension_vec
        forces[1:]  -= tension_vec
        
        # 3. Acceleration
        acc = forces / masses[:, None]
        acc[-1] = [0, 0]  # Rod tip fixed velocity
        
        return np.concatenate([vel.flatten(), acc.flatten()])

    # --- Initialization ---
    slack_factor = 1.0  # Start taut to avoid initial snap 
    init_x = np.linspace(0, tippet_length * slack_factor, segments + 1)
    init_y = np.full(segments + 1, river_depth)
    
    init_vx = np.full(segments + 1, surface_velocity)
    init_vx[-1] = surface_velocity * rod_tip_speed_factor
    init_vy = np.zeros(segments + 1)
    
    state0 = np.concatenate([
        np.stack([init_x, init_y], axis=1).flatten(),
        np.stack([init_vx, init_vy], axis=1).flatten()
    ])
    
    # --- Solver ---
    if isinstance(time_span, (tuple, list)):
        t_start, t_end = time_span
    else:
        t_start, t_end = 0, time_span
        
    t_eval = np.linspace(t_start, t_end, 100)
    
    pbar = tqdm(total=t_end-t_start, unit='s', desc="Simulating Drift")
    
    def progress_wrapper(t, y):
        pbar.n = min(t, t_end)
        pbar.refresh()
        return system_derivatives_vec(t, y)

    # Radau is essential for this high stiffness
    sol = solve_ivp(
        progress_wrapper, 
        (t_start, t_end), 
        state0, 
        method='Radau', 
        t_eval=t_eval,
        rtol=1e-2, 
        atol=1e-3
    )
    pbar.close()

    x = sol.y[0:2*segments+1:2]
    y = sol.y[1:2*segments+2:2]
    vx = sol.y[2*segments+2:4*segments+3:2]
    vy = sol.y[2*segments+3:4*segments+4:2]
    t = sol.t
    water_vx = ut.get_river_velocity(y, river_depth, surface_velocity, river_profile_exp)

    drift = {'x': x, 
             'y': y, 
             'vx': vx, 
             'vy': vy, 
             'water_vx': water_vx,
             't': t}

    return drift    