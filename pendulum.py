import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
import csv
import os

os.makedirs("pendulum", exist_ok=True)

# Simulation Data Path
SIM_DATA_FILE = "pendulum/data.csv"
GENERATE_DATA = True

# Pendulum parameters
INITIAL_ANGLE = np.deg2rad(120.0);
LENGTH = 1.0  # Length of pendulum (m)
GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)
DAMPING = 0.2 # Damping coefficient
DT = 0.015  # Time step (s)
NUM_FRAMES = 2500  # Number of frames to simulate
MASS = 1.0 # Mass of pendulum bob (only relevant for energy computation)
USD_PATH = "pendulum/pendulum.usd"

# Create CSV file and write header
if (GENERATE_DATA):
    csvfile = open(SIM_DATA_FILE, 'w', newline='')

    writer = csv.writer(csvfile)

    # Write system parameters as comments (first few rows)
    writer.writerow(['# SYSTEM PARAMETERS'])
    writer.writerow(['LENGTH', 'GRAVITY', 'DAMPING', 'DT', 'MASS', 'INITIAL_ANGLE', 'NUM_FRAMES'])
    writer.writerow([LENGTH, GRAVITY, DAMPING, DT, MASS, INITIAL_ANGLE, NUM_FRAMES])
    writer.writerow([])  # Empty row for separation

    # Write data column headers
    writer.writerow(['frame', 'time', 'theta', 'theta_dot', 'potential_energy', 'kinetic_energy', 'total_energy'])

# Initialize Warp
wp.init()

@wp.kernel
def integrate_pendulum(
    theta: wp.array(dtype=float),
    omega: wp.array(dtype=float),
    length: float,
    gravity: float,
    damping: float,
    dt: float
):
    # Simple pendulum equation: d²θ/dt² = -(g/L)sin(θ) - damping*dθ/dt
    angular_accel = -(gravity / length) * wp.sin(theta[0]) - damping * omega[0]
    
    # Update angular velocity and position
    omega[0] = omega[0] + angular_accel * dt
    theta[0] = theta[0] + omega[0] * dt

@wp.kernel
def update_transforms(
    theta: wp.array(dtype=float),
    transforms: wp.array(dtype=wp.transform),
    length: float
):
    # Pivot point (fixed)
    transforms[0] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    
    # Bob position
    x = length * wp.sin(theta[0])
    y = -length * wp.cos(theta[0])
    transforms[1] = wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity())

# Initialize state
theta = wp.array([INITIAL_ANGLE], dtype=float)  # Initial angle
omega = wp.array([0.0], dtype=float)  # Initial angular velocity

# Create model for rendering
builder = wp.sim.ModelBuilder()

# Add two bodies: pivot and bob
builder.add_body(origin=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()))
builder.add_body(origin=wp.transform([0.0, -LENGTH, 0.0], wp.quat_identity()))

# Add shapes for visualization
builder.add_shape_sphere(
    body=0,
    radius=0.05,
    density=100.0
)

builder.add_shape_sphere(
    body=1,
    radius=0.1,
    density=100.0
)

model = builder.finalize()
model.ground = False

# Create state
state = model.state()

# Position array for updating
transforms = wp.array(np.zeros((2, 7)), dtype=wp.transform)

# Setup renderer
renderer = wp.sim.render.SimRenderer(model, USD_PATH, scaling=1.0)

print(f"Simulating {NUM_FRAMES} frames...")

# Simulation loop
time = 0
for frame in range(NUM_FRAMES):
    theta_np = theta.numpy()
    omega_np = omega.numpy()
    height = LENGTH * (1.0 - np.cos(theta_np[0]))
    potential = MASS * GRAVITY * height

    linear_velocity = omega_np[0] * LENGTH
    kinetic = 0.5 * MASS * linear_velocity * linear_velocity
    
    total_energy = potential + kinetic

    if (GENERATE_DATA):
        # Write data row to CSV
        writer.writerow([frame, time, theta_np[0], omega_np[0], potential, kinetic, total_energy])
         
    # Integrate pendulum physics
    wp.launch(integrate_pendulum, dim=1, 
              inputs=[theta, omega, LENGTH, GRAVITY, DAMPING, DT])
    
    # Update body positions based on pendulum angle
    wp.launch(update_transforms, dim=1,
              inputs=[theta, transforms, LENGTH])
    
    # Copy transforms to state
    state.body_q.assign(transforms)

    # Update string positions for this frame
    transforms_cpu = transforms.numpy()
    pivot_pos = transforms_cpu[0][:3]  # First 3 values are position
    bob_pos = transforms_cpu[1][:3]
    
    # Render frame
    renderer.begin_frame(frame * DT)
    renderer.render(state)

    # Render string as a line
    renderer.render_line_strip(
        name="string",
        vertices=np.array([pivot_pos, bob_pos], dtype=np.float32),
        color=(0.8, 0.8, 0.8),
        radius=0.01
    )

    renderer.end_frame()
    
    if frame % 50 == 0:
        print(f"Frame {frame}/{NUM_FRAMES}")

    time = time + DT

# Save USD file
renderer.save()
