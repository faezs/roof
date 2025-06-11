#!/usr/bin/env python3
"""
Unit tests for heliostat math and physics
"""

import pytest
import jax.numpy as jnp
import numpy as np
from mujoco_heliostat_sim import (
    MuJoCoHeliostatSimulator, 
    HeliostatConfig,
    create_demo_heliostat_field,
    wrap_angle,
    angle_difference
)

class TestAngleMath:
    """Test basic angle math functions"""
    
    def test_wrap_angle(self):
        """Test angle wrapping function"""
        assert abs(wrap_angle(0.0) - 0.0) < 1e-6
        assert abs(wrap_angle(270.0) - (-90.0)) < 1e-6
        assert abs(wrap_angle(-270.0) - 90.0) < 1e-6
        assert abs(wrap_angle(450.0) - 90.0) < 1e-6
        # Note: 180° and -180° are equivalent, implementation can return either
        wrapped_180 = wrap_angle(180.0)
        assert wrapped_180 == 180.0 or wrapped_180 == -180.0
        wrapped_neg180 = wrap_angle(-180.0)
        assert wrapped_neg180 == 180.0 or wrapped_neg180 == -180.0
        
    def test_angle_difference(self):
        """Test angle difference calculation"""
        assert abs(angle_difference(10.0, 5.0) - 5.0) < 1e-6
        assert abs(angle_difference(5.0, 10.0) - (-5.0)) < 1e-6
        assert abs(angle_difference(350.0, 10.0) - (-20.0)) < 1e-6
        assert abs(angle_difference(10.0, 350.0) - 20.0) < 1e-6

class TestHeliostatConfig:
    """Test heliostat configuration"""
    
    def test_config_creation(self):
        """Test creating basic heliostat config"""
        config = HeliostatConfig(
            heliostat_id=0,
            position=jnp.array([0.0, 0.0, 0.0])
        )
        assert config.heliostat_id == 0
        assert config.max_torque == 50.0  # Should be our reduced value
        assert config.mirror_width == 2.0
        assert config.mirror_height == 1.0

class TestSunTrackingMath:
    """Test the actual sun tracking angle calculation"""
    
    def setup_method(self):
        """Set up test heliostat field"""
        self.heliostat_configs = create_demo_heliostat_field()
        
    def test_simple_tracking_case(self):
        """Test with simple, known values"""
        # Create minimal simulator just for angle calculation
        simulator = MuJoCoHeliostatSimulator(
            heliostat_configs=[HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))],
            timestep=0.01
        )
        
        # Simple case: sun from south, target to north
        sun_direction = jnp.array([0.0, -1.0, 0.0])  # From south
        target_position = jnp.array([0.0, 10.0, 0.0])  # To north
        
        angles = simulator.calculate_sun_tracking_angles(sun_direction, target_position)
        
        # Should get finite, reasonable angles
        assert jnp.isfinite(angles[0, 0])  # azimuth
        assert jnp.isfinite(angles[0, 1])  # elevation
        assert abs(angles[0, 0]) <= 180.0
        assert abs(angles[0, 1]) <= 90.0
        
    def test_demo_conditions_isolated(self):
        """Test with actual demo conditions but isolated"""
        simulator = MuJoCoHeliostatSimulator(
            heliostat_configs=self.heliostat_configs[:3],  # Just first 3
            timestep=0.01
        )
        
        # Demo conditions
        sun_direction = jnp.array([0.0, -0.7, -0.7])
        target_position = jnp.array([0.0, 50.0, 20.0])
        
        # This is where the overflow happens - let's see what values we get
        angles = simulator.calculate_sun_tracking_angles(sun_direction, target_position)
        
        print(f"Calculated angles: {angles}")
        
        # Check each heliostat
        for i in range(3):
            az, el = angles[i]
            print(f"Heliostat {i}: az={az:.2f}°, el={el:.2f}°")
            
            # These should all be finite
            assert jnp.isfinite(az), f"Azimuth not finite for heliostat {i}: {az}"
            assert jnp.isfinite(el), f"Elevation not finite for heliostat {i}: {el}"
            
            # Should be within reasonable ranges
            assert -180 <= az <= 180, f"Azimuth out of range: {az}"
            assert -90 <= el <= 90, f"Elevation out of range: {el}"

class TestControlSystem:
    """Test the control system math"""
    
    def test_control_torque_calculation(self):
        """Test PID control torque calculation"""
        simulator = MuJoCoHeliostatSimulator(
            heliostat_configs=[HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))],
            timestep=0.01
        )
        
        # Set a target
        simulator.set_target_angles(jnp.array([[45.0, 30.0]]))
        
        # Mock current states
        from mujoco_heliostat_sim import HeliostatState
        mock_states = [HeliostatState(
            azimuth=0.0,
            elevation=0.0, 
            azimuth_velocity=0.0,
            elevation_velocity=0.0,
            mirror_normal=jnp.array([0, 0, 1]),
            torques=jnp.array([0, 0])
        )]
        
        torques = simulator.compute_control_torques(mock_states)
        
        print(f"Control torques: {torques}")
        
        # Should get finite torques
        assert jnp.all(jnp.isfinite(torques))
        
        # Should be within max torque limits
        assert jnp.all(jnp.abs(torques) <= 50.0)  # Our max_torque

class TestPhysicsValues:
    """Test that physics values are reasonable"""
    
    def test_heliostat_field_positions(self):
        """Test that heliostat positions are reasonable"""
        configs = create_demo_heliostat_field()
        
        for i, config in enumerate(configs):
            pos = config.position
            print(f"Heliostat {i}: position {pos}")
            
            # Should be finite
            assert jnp.all(jnp.isfinite(pos))
            
            # Should be on ground (z=0)
            assert abs(pos[2]) < 1e-6
            
            # Should be within reasonable field bounds
            assert abs(pos[0]) <= 20.0
            assert abs(pos[1]) <= 20.0

def test_run_minimal_simulation():
    """Test running a minimal simulation to isolate the overflow"""
    
    # Single heliostat at origin
    config = HeliostatConfig(0, jnp.array([0.0, 0.0, 0.0]))
    
    simulator = MuJoCoHeliostatSimulator([config], timestep=0.01)
    
    # Simple target
    sun_direction = jnp.array([0.0, -1.0, -0.1])
    target_position = jnp.array([0.0, 10.0, 5.0])
    
    # Calculate angles - this should not overflow
    angles = simulator.calculate_sun_tracking_angles(sun_direction, target_position)
    print(f"Simple case angles: {angles}")
    
    # Set target
    simulator.set_target_angles(angles)
    
    # Try one simulation step
    initial_mjx_data = simulator.mjx_data
    print(f"Initial qpos: {initial_mjx_data.qpos}")
    print(f"Initial qvel: {initial_mjx_data.qvel}")
    
    # Get states
    states = simulator.get_heliostat_states(initial_mjx_data)
    print(f"Initial state: az={states[0].azimuth}, el={states[0].elevation}")
    
    # Compute torques
    torques = simulator.compute_control_torques(states)
    print(f"Control torques: {torques}")
    
    # Take one step - this is where overflow likely happens
    try:
        new_mjx_data = simulator.step(initial_mjx_data, torques)
        print(f"After step qpos: {new_mjx_data.qpos}")
        print(f"After step qvel: {new_mjx_data.qvel}")
        
        # Check for overflow
        assert jnp.all(jnp.isfinite(new_mjx_data.qpos))
        assert jnp.all(jnp.isfinite(new_mjx_data.qvel))
        
    except Exception as e:
        print(f"Simulation step failed: {e}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])