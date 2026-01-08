# Fly Drift Simulation

This project simulates the dynamics of the tippet and nymph fished euro-style in a river current. It uses a physics-based approach to model the interaction between the water flow, the tippet and nymph.

## Physics of Drift

As clearly demonstrated in [Modern Nymphing: European Inspired Techniques](https://tacticalflyfisher.com/products/modern-nymphing-european-inspired-techniques-dvd-featuring-devin-olsen-and-lance-egan) by Lance Egan and Devin Olsen, the surface water flows much faster than the bottom water. This impacts the drift of a nymph close to the bottom.

The drift of a nymph is governed by the competition between the local forces on the fly (which try to move it at the speed of the water surrounding the fly) and the drag forces on the line (which is continuously pulled downstream by the faster surface currents).

* **Drift Velocity**: In scenarios when the nymph is not corrected and drifted "dead drift", the nymph will drift faster than the bottom water around it. This is because the tippet pass through the higher-velocity upper water column, acting like a sail that pulls the nymph downstream.

* **Influence of Parameters**:

* *Nymph Weight (0.1 - 0.5g)*: Heavier nymphs sink faster, reaching the slower bottom layer sooner. Once there, their inertia and potential friction with the riverbed (ticking) help retard the downstream pull of the line.
* *Tippet Diameter (0.1–0.3mm)*: Thicker lines have more surface area ($A_{projected} = \text{diameter} \times \text{length}$), experiencing significantly higher drag from the fast surface water. This increases the unnatural downstream speed of the nymph. Thin tippets are critical for achieving a true dead drift that matches the bottom speed.
* *Fly Drag*: A "bushy" fly has a high drag coefficient ($C_D$). While this helps it move with the local water, it also creates more resistance against sinking, keeping the fly in faster water for longer.

## Mathematical Model

To simulate this, we treat the system as a **Tethered Underwater System**, similar to the cable-towing models described in the literature (e.g., Dowling, Spolek), but with the "towing" force replaced by the distributed drag of the river current.

### The River Velocity Profile

Rivers do not flow at a uniform speed. Friction with the riverbed causes the water to slow down significantly near the bottom. This is modeled using the **Logarithmic Law of the Wall**:

$$u(y) = \frac{u_*}{\kappa} \ln\left(\frac{y}{z_0}\right)$$

* $u(y)$: Water velocity at height $y$ from the bottom.
* $u_*$: Shear velocity (related to river slope and depth).
* $\kappa$: Von Kármán constant ($\approx 0.41$).
* $z_0$: Roughness height (determined by rock size on the riverbed).

#### Simplification for Simulation

A power-law approximation is often computationally faster and sufficient:

$$u(y) = U_{surface} \left( \frac{y}{h} \right)^{1/6}$$

where $h$ is the total river depth.

### Forces on the Nymph

The nymph is modeled as a point mass or small sphere subjected to three forces:

**Gravity ($F_g$):** Pulls downwards. $F_g = m_{fly} g$.

**Buoyancy ($F_b$):** Pushes upwards. $F_b = \rho_{water} V_{fly} g$.

**Hydrodynamic Drag ($F_{d,fly}$):** Acts opposite to the relative velocity between the water and the fly.

$$F_{d,fly} = \frac{1}{2} \rho_{water} C_{D,fly} A_{fly} |u(y_{fly}) - v_{fly}| (u(y_{fly}) - v_{fly})$$

### Forces on the Tippet

The tippet is a flexible cylinder spanning from the surface to the fly. It experiences **distributed drag** that varies with depth. Following the Morison Equation approach seen in cable dynamics literature (e.g., Dowling, Part 1), the force per unit length is split into normal and tangential components.

Because the line is nearly vertical, the **Normal Drag ($F_n$)** is the dominant force pushing the line downstream:

$$F_n(y) = \frac{1}{2} \rho_{water} C_{D,line} d_{line} (u(y) - v_{line,x}(y))^2$$

* $d_{line}$: Tippet diameter (crucial parameter).
* $u(y)$: River velocity at depth $y$ (fast at surface, slow at bottom).

**Key Insight:** Since $u(y)$ is large near the surface, $F_n$ is large at the top of the line, creating a strong moment that pulls the bottom nymph forward.

## Simulation Method: Lumped Mass Model

The most robust way to simulate this is the **Lumped Mass Method**. The line is divided into $N$ small segments (nodes).

### Steps

1. **Discretization**: Divide the tippet into $N$ segments connected by massless springs (representing elasticity) or rigid rods (if inextensible).
2. **State Vector**: Define the position $(x_i, y_i)$ and velocity $(\dot{x}_i, \dot{y}_i)$ for every node $i$. Node 0 is the fly; Node $N$ is the rod tip/indicator at the surface.
3. **Force Calculation Loop**:

* For each node $i$ at height $y_i$, calculate the local water velocity $u(y_i)$.
* Calculate Drag Force on the node based on $u(y_i) - \dot{x}_i$.
* Calculate Tension forces from neighbor nodes ($i-1$ and $i+1$) based on Hooke’s Law: $T = k(|\vec{r}_{i+1} - \vec{r}_i| - L_{seg})$.
* Add Gravity/Buoyancy.

1. **Time Stepping**: Use a numerical integrator (e.g., Runge-Kutta 4 or Euler) to update positions:

$$\vec{a}_i = \frac{\sum \vec{F}_{acting\_on\_i}}{m_i}$$$$\vec{v}_{i, t+\Delta t} = \vec{v}_{i, t} + \vec{a}_i \Delta t$$$$\vec{p}_{i, t+\Delta t} = \vec{p}_{i, t} + \vec{v}_{i, t+\Delta t} \Delta t$$

## Files

* `simulation.py`: Contains the core physics engine and class definitions (`RiverFlow`, `Node`, `Line`).
* `utils.py`: Contains utility functions for the simulation.
* `visualize.py`: Plots the simulation results and generates animated GIFs.

## Dependencies

* Python 3.x
* `numpy`
* `matplotlib`
* `scipy`
* `tqdm` (for progress bar)

## Usage

This project is designed to be used as a Python library where you import the simulation and visualization modules into your own scripts.

### 1. Running a Simulation

Use the `simulate` function from `simulate.py`. You can customize parameters like fly mass, tippet diameter, and rod handling.

### 2. Visualizing Results

The `visualize.py` module provides functions to plot static graphs (`plot_drift`) or generate animations (`make_movie`).

### Example Script

Here is a complete example of how to run a simulation and visualize the result:

```python
import matplotlib.pyplot as plt
from simulate import simulate
from visualize import plot_drift, make_movie
from utils import calculate_sink_metrics

# --- 1. Run the Simulation ---
# Returns a dictionary containing position (x, y), velocity, and time data
drift_data = simulate(
    fly_mass='3.0mm',           # Fly bead size (or float in kg)
    tippet_diameter='6x',       # Tippet diameter (or float in m)
    bugginess=1.0,              # 1-4 scale of fly drag/bushiness
    tippet_length=1.5,          # Length of the tippet in meters
    rod_tip_speed_factor=0.85,  # Speed of rod tip relative to surface water (1.0 = dead drift)
    river_depth=1.5,            # Depth of the river in meters
    surface_velocity=1.5,       # Surface velocity in m/s
    river_profile_exp=0.166,    # Power-law exponent for velocity profile
    time_span=15.0,             # Duration of simulation in seconds
    segments=20                 # Number of segments in the line model
)

# --- 2. Visualize Results ---

# A. Print Metrics
metrics = calculate_sink_metrics(drift_data)
print(f"Max Drift Depth: {metrics['max_depth']:.2f} m")
print(f"Average Sink Rate: {metrics['average_sink_rate']:.2f} m/s")

# B. Static Plot
fig, ax = plt.subplots(figsize=(12, 6))
plot_drift(drift_data, ax=ax, title="Nymph Drift Profile")
plt.savefig("drift_result.png")
print("Saved drift_result.png")

# C. Create an Animation (GIF)
# Note: Requires ffmpeg or pillow
make_movie(
    drifts=[drift_data], 
    filename='drift_animation.gif', 
    fps=15, 
    title=f"Drift Simulation (Max Depth: {metrics['max_depth']:.2f}m)"
)
```

## Simulation Principles & Physics

The simulation relies on a discretized mass-spring-damper like system, refined with **Position Based Dynamics (PBD)** constraints for stability and inextensibility.

### 1. River Flow Model

The river flow is modeled using a power-law velocity profile, which is common in open-channel flow hydraulics:
$$ v(y) = v_{surface} \cdot \left( \frac{y}{depth} \right)^{1/4} $$

* **v(y)**: Velocity at depth y.
* **v_surface**: Surface velocity in m/s.
* **y**: Distance from the river bottom.
* **depth**: Total depth of the river.

### 2. Tippet & Fly Physics

The tippet is approximated as a chain of connected **Nodes** (point masses).

* **Gravity**: Applied to all nodes ($F = mg$).
* **Buoyancy**: Archimedes' principle is applied based on the volume of the line segments and the density of water.
    $$ F_b = V_{segment} \cdot \rho_{water} \cdot g $$
* **Drag (Hydrodynamics)**: A semi-implicit drag formulation is used to simulate fluid resistance. The drag force is proportional to the square of the relative velocity between the water and the line node.
    $$ F_{drag} \propto C_d \cdot A \cdot |v_{rel}|^2 $$
  * *Note*: The simulation distinguishes between submerged segments (drag applied) and segments in the air (no drag/ballistic motion).
* **Constraint Solving (PBD)**: Instead of using stiff springs which can be unstable, the length of line segments is enforced using an iterative constraint solver. This ensures the line behaves like a rope (inextensible) rather than a rubber band.

### Collision Handling

* **River Bed**: Inelastic collision with the bottom ($y=0$). Friction is simulated by reducing horizontal velocity upon contact.
