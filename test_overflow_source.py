#!/usr/bin/env python3
"""
Isolate the exact source of the overflow warning
"""

import warnings
import numpy as np
import jax.numpy as jnp
from mujoco_heliostat_sim import MuJoCoHeliostatSimulator, HeliostatConfig

def test_overflow_source():
    """Turn warnings into errors to get stack trace"""
    
    # Turn warnings into errors to get stack trace
    warnings.filterwarnings('error', category=RuntimeWarning, message='.*overflow encountered in cast.*')
    
    print("Creating minimal heliostat...")
    config = HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))
    
    try:
        print("Initializing simulator...")
        simulator = MuJoCoHeliostatSimulator([config], timestep=0.01)
        print("✓ Simulator created without overflow")
        
        print("Testing angle calculation...")
        sun_direction = jnp.array([0.0, -1.0, -0.1])
        target_position = jnp.array([0.0, 10.0, 5.0])
        angles = simulator.calculate_sun_tracking_angles(sun_direction, target_position)
        print(f"✓ Angles calculated: {angles}")
        
        print("Setting target angles...")
        simulator.set_target_angles(angles)
        print("✓ Target angles set")
        
        print("Getting initial state...")
        states = simulator.get_heliostat_states(simulator.mjx_data)
        print(f"✓ Initial state: az={states[0].azimuth}, el={states[0].elevation}")
        
        print("Computing control torques...")
        torques = simulator.compute_control_torques(states)
        print(f"✓ Torques: {torques}")
        
        print("Taking simulation step - THIS IS WHERE OVERFLOW LIKELY HAPPENS...")
        new_mjx_data = simulator.step(simulator.mjx_data, torques)
        print("✓ Simulation step completed without overflow")
        
    except RuntimeWarning as e:
        print(f"CAUGHT OVERFLOW WARNING: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's see what values are causing the overflow
        print("\nDebugging values that might cause overflow:")
        print(f"Torques shape: {torques.shape}, values: {torques}")
        print(f"Initial qpos: {simulator.mjx_data.qpos}")
        print(f"Initial qvel: {simulator.mjx_data.qvel}")
        
        # Check for any extremely large values
        max_torque = jnp.max(jnp.abs(torques))
        max_qpos = jnp.max(jnp.abs(simulator.mjx_data.qpos))
        max_qvel = jnp.max(jnp.abs(simulator.mjx_data.qvel))
        
        print(f"Max torque magnitude: {max_torque}")
        print(f"Max qpos magnitude: {max_qpos}")
        print(f"Max qvel magnitude: {max_qvel}")
        
        # Check data types
        print(f"Torques dtype: {torques.dtype}")
        print(f"qpos dtype: {simulator.mjx_data.qpos.dtype}")
        print(f"qvel dtype: {simulator.mjx_data.qvel.dtype}")

def test_step_by_step():
    """Test each part of the simulation step individually"""
    
    print("\n=== STEP BY STEP DEBUGGING ===")
    
    config = HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))
    simulator = MuJoCoHeliostatSimulator([config], timestep=0.01)
    
    # Very simple torques
    simple_torques = jnp.array([[1.0, 1.0]])
    
    print(f"Testing with simple torques: {simple_torques}")
    
    try:
        # Test just the JIT function directly
        from mujoco_heliostat_sim import _mjx_step_with_forces
        
        print("Testing JIT function directly...")
        result = _mjx_step_with_forces(simulator.mjx_model, simulator.mjx_data, simple_torques)
        print("✓ JIT function completed")
        
    except Exception as e:
        print(f"JIT function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_overflow_source()
    test_step_by_step()