# SolTrace API Cheatsheet

This document provides a quick reference guide for using the SolTrace Python API (`pysoltrace`) for optical modeling of heliostats and other solar concentrators.

## Basic Setup

```python
import sys
sys.path.append("/path/to/soltrace/lib")
import pysoltrace
from pysoltrace import PySolTrace, Point

# Create API instance
sim = PySolTrace()
```

## Creating Optical Properties

```python
# Create reflective surface (mirror)
mirror = sim.add_optic("Mirror")
mirror.front.reflectivity = 0.95       # 95% reflectivity
mirror.front.slope_error = 2.0         # 2 mrad slope error
mirror.front.specularity = 0.999       # Highly specular surface

# Create absorptive surface
absorber = sim.add_optic("Absorber")
absorber.front.reflectivity = 0.1      # 10% reflectivity (90% absorption)
absorber.back.reflectivity = 0.0       # No reflection from back side
```

## Setting Up the Sun

```python
# Create sun
sun = sim.add_sun()

# Point source at infinite distance (default)
sun.point_source = False

# Sun shape: 'p' for pillbox, 'g' for gaussian, 'd' for data table
sun.shape = 'p'  
sun.sigma = 4.65  # mrad (pillbox half-angle or gaussian standard deviation)

# Set sun position (direction vector for incoming rays)
sun.position.x = 0
sun.position.y = 0
sun.position.z = 1   # Directly overhead
```

## Creating Stages and Elements

```python
# Create a new stage
stage = sim.add_stage()

# Add a flat mirror element
mirror_element = stage.add_element()
mirror_element.optic = mirror  # Use the optic defined earlier
mirror_element.position = Point(0, 0, 5)  # Position (meters)
mirror_element.aim = Point(0, 0, 0)       # Aim point (meters)
mirror_element.zrot = 0                   # Rotation around z-axis (degrees)

# Set element geometry
mirror_element.surface_flat()                  # Flat surface
# Or for curved:
# mirror_element.surface_parabolic(f=10)       # Parabolic with 10m focal length
# mirror_element.surface_spherical(rad=20)     # Spherical with 20m radius

# Set aperture shape
mirror_element.aperture_rectangle(2, 1)        # 2m width x 1m height
# Or:
# mirror_element.aperture_circle(radius=1)     # 1m radius circle
```

## Heliostat Configuration Example

```python
# Create a heliostat
heliostat_stage = sim.add_stage()
heliostat = heliostat_stage.add_element()
heliostat.optic = mirror
heliostat.position = Point(10, 0, 0)  # 10m east of origin

# Target is at the origin, 20m high
target = Point(0, 0, 20)

# Calculate aim vector (bisector of sun and target directions)
to_target = target - heliostat.position
to_target = to_target.unitize()

sun_vec = sun.position.unitize()

# Calculate the reflection vector (Law of reflection)
aim = (to_target + sun_vec).unitize()
heliostat.aim = heliostat.position + aim * 100  # Extend vector

# Calculate proper z-rotation for heliostat
heliostat.zrot = sim.util_calc_zrot_azel(aim)

# Set element geometry
heliostat.surface_flat()  
heliostat.aperture_rectangle(2, 2)  # 2m x 2m square
```

## Running Simulation

```python
# Set ray tracing parameters
sim.num_ray_hits = 100000       # Minimum number of successful ray hits
sim.max_rays_traced = 1000000   # Maximum number of rays to trace
sim.is_sunshape = True          # Include sunshape in ray generation
sim.is_surface_errors = True    # Include surface errors

# Run the simulation
# Must be in __main__ guard for multithreading
if __name__ == "__main__":
    sim.run(seed=-1, as_power_tower=True, nthread=8)
```

## Analysis and Visualization

```python
# Get ray data
print(f"Number of rays traced: {sim.raydata.index.size}")

# Create 3D visualization of ray traces
sim.plot_trace(nrays=10000, ntrace=100)

# Create flux map on a specific element (e.g., receiver)
sim.plot_flux(
    element=receiver_element,  
    nx=50,                      # Number of bins in x direction
    ny=50,                      # Number of bins in y direction
    figpath="flux_map.png",     # Save to file
    display=True                # Show the plot
)
```

## Saving/Loading Models

```python
# Save model to SolTrace input file
sim.write_soltrace_input_file("heliostat_model.stinput")

# Create a new instance and load from file
new_sim = PySolTrace()
# Load file using SolTrace application
```

## Utilities

```python
# Calculate Euler angles for positioning
euler_angles = sim.util_calc_euler_angles(
    origin=Point(0, 0, 0),
    aimpoint=Point(0, 0, 1),
    zrot=0
)

# Transform coordinates between reference and local systems
transform = sim.util_calc_transforms(euler_angles)
local_coords = sim.util_transform_to_local(
    posref=numpy.array([1, 2, 3]),
    cosref=numpy.array([0, 0, 1]),
    origin=numpy.array([0, 0, 0]),
    rreftoloc=transform["rreftoloc"]
)
```

## Common Heliostat Field Patterns

```python
# Create circular field of heliostats
num_heliostats = 8
radius = 20  # meters

for i in range(num_heliostats):
    angle = i * 2 * math.pi / num_heliostats
    pos_x = radius * math.sin(angle)
    pos_y = radius * math.cos(angle)
    
    heliostat = stage.add_element()
    heliostat.optic = mirror
    heliostat.position = Point(pos_x, pos_y, 0)
    
    # Calculate aim as above...
```

## Important Notes

1. All angles in the API are in degrees unless otherwise specified
2. All distances are in meters
3. The coordinate system is right-handed:
   - X: East
   - Y: North
   - Z: Up
4. When using multithreading, always put `sim.run()` inside a `if __name__ == "__main__"` guard
5. The `plot_trace()` function requires the plotly package to be installed