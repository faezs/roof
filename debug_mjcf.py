#!/usr/bin/env python3
"""
Debug the MJCF model values to find the overflow source
"""

from mujoco_heliostat_sim import MuJoCoHeliostatSimulator, HeliostatConfig
import jax.numpy as jnp

def debug_mjcf_values():
    """Check what values are being generated in the MJCF"""
    
    config = HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))
    
    print("=== Heliostat Config Values ===")
    print(f"Mirror mass: {config.mirror_mass}")
    print(f"Mirror width: {config.mirror_width}")  
    print(f"Mirror height: {config.mirror_height}")
    print(f"Max torque: {config.max_torque}")
    print(f"Pedestal height: {config.pedestal_height}")
    
    # Calculate inertia values that would be generated
    mass = config.mirror_mass
    inertia_x = mass * 0.1
    inertia_y = mass * 0.1  
    inertia_z = mass * 0.05
    
    print(f"\n=== Generated Inertia Values ===")
    print(f"Mass: {mass}")
    print(f"Inertia X: {inertia_x}")
    print(f"Inertia Y: {inertia_y}")
    print(f"Inertia Z: {inertia_z}")
    
    # Check if any are extremely large
    values_to_check = [mass, inertia_x, inertia_y, inertia_z, config.max_torque, 
                       config.mirror_width, config.mirror_height, config.pedestal_height]
    
    print(f"\n=== Value Range Check ===")
    for i, val in enumerate(values_to_check):
        print(f"Value {i}: {val}, float32 range? {-3.4e38 < val < 3.4e38}")
        
    # Let's also look at the actual XML being generated
    simulator = MuJoCoHeliostatSimulator([config], timestep=0.01)
    
    # Read the generated XML file
    with open(simulator.mjcf_path, 'r') as f:
        xml_content = f.read()
    
    print(f"\n=== Generated MJCF Content ===")
    print("Looking for potentially problematic values...")
    
    lines = xml_content.split('\n')
    for i, line in enumerate(lines):
        if 'inertial' in line or 'mass' in line or 'diaginertia' in line:
            print(f"Line {i+1}: {line.strip()}")
            
        # Look for very large numbers
        import re
        large_numbers = re.findall(r'\d+\.?\d*e[+-]?\d+|\d{10,}', line)
        if large_numbers:
            print(f"Line {i+1} has large numbers: {line.strip()}")
            print(f"  Large numbers found: {large_numbers}")

if __name__ == "__main__":
    debug_mjcf_values()