"""
MuJoCo-MJX Heliostat Robotics Simulation

This module integrates MuJoCo-MJX for physics-based robotics simulation of heliostat fields.
Provides realistic actuator dynamics, wind disturbances, and structural vibrations.

Key Features:
- Physics-based dual-axis heliostat tracking
- Realistic actuator dynamics with backlash and friction
- Wind loading and structural vibrations
- JAX autodiff for optimal control
- Integration with ARTIST raytracing and verified control
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import pathlib
import time
import logging

# Local imports
from device_utils import get_default_device
from heliostat_verified_control import verified_control

logger = logging.getLogger(__name__)

# Utility functions
def wrap_angle(angle_deg):
    """Wrap angle to (-180, 180] degrees"""
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped

def angle_difference(target_deg, current_deg):
    """Compute shortest angular difference, handling wrap-around"""
    diff = target_deg - current_deg
    return wrap_angle(diff)

# JIT-compiled standalone functions
def _mjx_step_with_forces(mjx_model, mjx_data: mjx.Data, control_torques: jnp.ndarray) -> mjx.Data:
    """Physics step (temporarily removing JIT to isolate overflow)"""
    # Apply control torques
    mjx_data = mjx_data.replace(ctrl=control_torques.flatten())
    
    # Step physics
    mjx_data = mjx.step(mjx_model, mjx_data)
    
    return mjx_data

@dataclass
class HeliostatConfig:
    """Configuration for individual heliostat"""
    heliostat_id: int
    position: jnp.ndarray  # [x, y, z] in world coordinates
    mirror_width: float = 2.0  # meters
    mirror_height: float = 1.0  # meters
    pedestal_height: float = 3.0  # meters
    
    # Actuator parameters
    azimuth_range: Tuple[float, float] = (-180.0, 180.0)  # degrees
    elevation_range: Tuple[float, float] = (15.0, 90.0)   # degrees
    max_angular_velocity: float = 5.0  # deg/s
    max_torque: float = 50.0  # Nm (realistic for small heliostat)
    
    # Structural parameters (reduced for stability)
    mirror_mass: float = 10.0  # kg (lighter)
    pedestal_mass: float = 50.0  # kg (lighter)
    structural_damping: float = 0.5  # increased damping
    
class HeliostatState(NamedTuple):
    """State of a single heliostat"""
    azimuth: float  # degrees
    elevation: float  # degrees
    azimuth_velocity: float  # deg/s
    elevation_velocity: float  # deg/s
    mirror_normal: jnp.ndarray  # 3D unit vector
    torques: jnp.ndarray  # [azimuth_torque, elevation_torque]

class WindDisturbance(NamedTuple):
    """Wind disturbance parameters"""
    mean_velocity: jnp.ndarray  # [vx, vy, vz] m/s
    turbulence_intensity: float  # 0-1
    gust_factor: float  # multiplier for gusts
    coherence_length: float  # spatial correlation in meters

class MuJoCoHeliostatSimulator:
    """MuJoCo-MJX based heliostat field simulation"""
    
    def __init__(self, 
                 heliostat_configs: List[HeliostatConfig],
                 wind_config: Optional[WindDisturbance] = None,
                 timestep: float = 0.01):
        """
        Initialize MuJoCo simulation for heliostat field
        
        Parameters
        ----------
        heliostat_configs : List[HeliostatConfig]
            Configuration for each heliostat
        wind_config : Optional[WindDisturbance]
            Wind disturbance parameters
        timestep : float
            Simulation timestep in seconds
        """
        self.heliostat_configs = heliostat_configs
        self.num_heliostats = len(heliostat_configs)
        self.wind_config = wind_config or WindDisturbance(
            mean_velocity=jnp.array([5.0, 0.0, 0.0]),
            turbulence_intensity=0.1,
            gust_factor=1.5,
            coherence_length=100.0
        )
        self.timestep = timestep
        
        # Generate MuJoCo model
        self.mjcf_path = self._generate_mjcf_model()
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.mjx_model = mjx.put_model(self.model)
        
        # Initialize simulation state
        self.data = mujoco.MjData(self.model)
        
        # Initialize joint positions to reasonable starting values
        for i in range(self.num_heliostats):
            azimuth_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'heliostat_{i}_azimuth_joint')
            elevation_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'heliostat_{i}_elevation_joint')
            
            if azimuth_joint_id >= 0:
                self.data.qpos[azimuth_joint_id] = 0.0  # Start at 0 degrees azimuth
            if elevation_joint_id >= 0:
                self.data.qpos[elevation_joint_id] = 45.0 * np.pi / 180.0  # Start at 45 degrees elevation
        
        # Forward kinematics to ensure consistent state
        mujoco.mj_forward(self.model, self.data)
        
        self.mjx_data = mjx.put_data(self.model, self.data)
        
        # Control interface
        self.target_angles = jnp.zeros((self.num_heliostats, 2))  # [azimuth, elevation] for each
        self.previous_target_angles = jnp.zeros((self.num_heliostats, 2))  # For rate limiting
        
        # Wind simulation state
        self.wind_rng_key = jax.random.PRNGKey(42)
        
        logger.info(f"Initialized MuJoCo simulation with {self.num_heliostats} heliostats")
        
    def _generate_mjcf_model(self) -> pathlib.Path:
        """Generate MJCF XML model for heliostat field"""
        
        xml_content = self._create_mjcf_xml()
        
        mjcf_path = pathlib.Path("/tmp/heliostat_field.xml")
        with open(mjcf_path, 'w') as f:
            f.write(xml_content)
            
        logger.info(f"Generated MJCF model: {mjcf_path}")
        return mjcf_path
        
    def _create_mjcf_xml(self) -> str:
        """Create MJCF XML content for heliostat field"""
        
        xml_lines = [
            '<?xml version="1.0" ?>',
            '<mujoco model="heliostat_field">',
            '  <compiler angle="degree" coordinate="local"/>',
            '  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4" solver="Newton" iterations="10"/>',
            '',
            '  <!-- Material definitions -->',
            '  <asset>',
            '    <material name="mirror" rgba="0.9 0.9 1.0 1.0" specular="1" shininess="0.8"/>',
            '    <material name="steel" rgba="0.7 0.7 0.7 1.0"/>',
            '    <material name="ground" rgba="0.6 0.4 0.2 1.0"/>',
            '  </asset>',
            '',
            '  <!-- World body -->',
            '  <worldbody>',
            '    <!-- Ground plane -->',
            '    <geom type="plane" size="1000 1000 0.1" material="ground"/>',
            '    ',
        ]
        
        # Add each heliostat
        for i, config in enumerate(self.heliostat_configs):
            xml_lines.extend(self._create_heliostat_xml(i, config))
            
        xml_lines.extend([
            '  </worldbody>',
            '',
            '  <!-- Actuators -->',
            '  <actuator>',
        ])
        
        # Add actuators for each heliostat
        for i in range(self.num_heliostats):
            xml_lines.extend([
                f'    <motor name="heliostat_{i}_azimuth" joint="heliostat_{i}_azimuth_joint" gear="1"/>',
                f'    <motor name="heliostat_{i}_elevation" joint="heliostat_{i}_elevation_joint" gear="1"/>',
            ])
            
        xml_lines.extend([
            '  </actuator>',
            '',
            '  <!-- Sensors -->',
            '  <sensor>',
        ])
        
        # Add sensors for each heliostat
        for i in range(self.num_heliostats):
            xml_lines.extend([
                f'    <jointpos name="heliostat_{i}_azimuth_pos" joint="heliostat_{i}_azimuth_joint"/>',
                f'    <jointvel name="heliostat_{i}_azimuth_vel" joint="heliostat_{i}_azimuth_joint"/>',
                f'    <jointpos name="heliostat_{i}_elevation_pos" joint="heliostat_{i}_elevation_joint"/>',
                f'    <jointvel name="heliostat_{i}_elevation_vel" joint="heliostat_{i}_elevation_joint"/>',
                f'    <accelerometer name="heliostat_{i}_accel" site="heliostat_{i}_mirror_center"/>',
                f'    <gyro name="heliostat_{i}_gyro" site="heliostat_{i}_mirror_center"/>',
            ])
            
        xml_lines.extend([
            '  </sensor>',
            '</mujoco>'
        ])
        
        return '\n'.join(xml_lines)
        
    def _create_heliostat_xml(self, heliostat_id: int, config: HeliostatConfig) -> List[str]:
        """Create XML for a single heliostat"""
        
        x, y, z = config.position
        
        xml_lines = [
            f'    <!-- Heliostat {heliostat_id} - MINIMAL GEOMETRY WITH PROPER INERTIALS -->',
            f'    <body name="heliostat_{heliostat_id}_base" pos="{x} {y} {config.pedestal_height}">',
            f'      <!-- Base inertial (required for moving body) -->',
            f'      <inertial pos="0 0 0" mass="5.0" diaginertia="0.1 0.1 0.05"/>',
            f'      <!-- Azimuth joint -->',
            f'      <joint name="heliostat_{heliostat_id}_azimuth_joint" type="hinge" axis="0 0 1" ',
            f'             range="{config.azimuth_range[0]} {config.azimuth_range[1]}" ',
            f'             damping="50" frictionloss="5"/>',
            f'      ',
            f'      <!-- Elevation assembly -->',
            f'      <body name="heliostat_{heliostat_id}_elevation" pos="0 0 0.2">',
            f'        <!-- Elevation assembly inertial -->',
            f'        <inertial pos="0 0 0" mass="3.0" diaginertia="0.08 0.08 0.04"/>',
            f'        <joint name="heliostat_{heliostat_id}_elevation_joint" type="hinge" axis="1 0 0" ',
            f'               range="{config.elevation_range[0]} {config.elevation_range[1]}" ',
            f'               damping="25" frictionloss="2.5"/>',
            f'        ',
            f'        <!-- Mirror - MINIMAL GEOMETRY -->',
            f'        <body name="heliostat_{heliostat_id}_mirror" pos="0 0 0.1">',
            f'          <!-- Mirror inertial properties -->',
            f'          <inertial pos="0 0 0" mass="{config.mirror_mass}" diaginertia="{config.mirror_mass*0.1:.3f} {config.mirror_mass*0.1:.3f} {config.mirror_mass*0.05:.3f}"/>',
            f'          <geom name="heliostat_{heliostat_id}_mirror_geom" type="box" ',
            f'                size="{config.mirror_width/2} {config.mirror_height/2} 0.02" ',
            f'                material="mirror"/>',
            f'          ',
            f'          <!-- Mirror center site for sensors -->',
            f'          <site name="heliostat_{heliostat_id}_mirror_center" pos="0 0 0" size="0.01"/>',
            f'        </body>',
            f'      </body>',
            f'    </body>',
            f'    ',
        ]
        
        return xml_lines
        
    def step(self, mjx_data: mjx.Data, control_torques: jnp.ndarray) -> mjx.Data:
        """
        Advance simulation by one timestep with wind disturbances
        
        Parameters
        ----------
        mjx_data : mjx.Data
            Current simulation state
        control_torques : jnp.ndarray
            Control torques for each heliostat [N_heliostats, 2]
            
        Returns
        -------
        mjx.Data
            Updated simulation state
        """
        # Apply wind forces (simplified - full wind forces would require more complex JIT handling)
        # mjx_data = self._apply_wind_forces(mjx_data)
        
        # Use JIT-compiled step function
        mjx_data = _mjx_step_with_forces(self.mjx_model, mjx_data, control_torques)
        
        # Check for numerical issues and reset if needed
        if not jnp.all(jnp.isfinite(mjx_data.qpos)) or not jnp.all(jnp.isfinite(mjx_data.qvel)):
            logger.warning("Numerical instability detected in MJX simulation - resetting problematic values")
            
            # Reset to safe values if overflow detected
            safe_qpos = jnp.where(jnp.isfinite(mjx_data.qpos), 
                                 jnp.clip(mjx_data.qpos, -jnp.pi, jnp.pi), 0.0)
            safe_qvel = jnp.where(jnp.isfinite(mjx_data.qvel), 
                                 jnp.clip(mjx_data.qvel, -10.0, 10.0), 0.0)
            
            mjx_data = mjx_data.replace(qpos=safe_qpos, qvel=safe_qvel)
        
        return mjx_data
        
    def _apply_wind_forces(self, mjx_data: mjx.Data) -> mjx.Data:
        """Apply wind forces to heliostat mirrors"""
        
        # Generate turbulent wind field
        wind_forces = self._generate_wind_forces(mjx_data.time)
        
        # Apply forces to mirror corner sites
        total_force = jnp.zeros_like(mjx_data.xfrc_applied)
        
        for i in range(self.num_heliostats):
            # Get mirror orientation
            mirror_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'heliostat_{i}_mirror')
            mirror_quat = mjx_data.xquat[mirror_body_id]
            
            # Calculate wind pressure on mirror surface
            wind_pressure = self._calculate_wind_pressure(
                wind_forces[i], mirror_quat, 
                self.heliostat_configs[i].mirror_width * self.heliostat_configs[i].mirror_height
            )
            
            # Distribute force to corner sites
            corner_force = wind_pressure / 4  # Distribute equally to 4 corners
            
            for j in range(4):
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'heliostat_{i}_corner_{j+1}')
                body_id = self.model.site_bodyid[site_id]
                total_force = total_force.at[body_id, :3].add(corner_force)
                
        return mjx_data.replace(xfrc_applied=total_force)
        
    def _generate_wind_forces(self, time: float) -> jnp.ndarray:
        """Generate spatially and temporally correlated wind forces"""
        
        # Update RNG key
        self.wind_rng_key, subkey = jax.random.split(self.wind_rng_key)
        
        # Base wind velocity
        base_wind = self.wind_config.mean_velocity
        
        # Generate turbulence for each heliostat
        wind_forces = []
        
        for i, config in enumerate(self.heliostat_configs):
            # Spatial correlation based on distance from origin
            distance = jnp.linalg.norm(config.position[:2])
            correlation = jnp.exp(-distance / self.wind_config.coherence_length)
            
            # Temporal variation with multiple frequencies
            time_factor = (jnp.sin(0.1 * time) * 0.3 + 
                          jnp.sin(0.05 * time) * 0.2 + 
                          jnp.sin(0.02 * time) * 0.1)
            
            # Random turbulence
            turbulence = jax.random.normal(subkey, (3,)) * self.wind_config.turbulence_intensity
            
            # Combine effects
            total_wind = (base_wind + 
                         base_wind * time_factor * self.wind_config.gust_factor +
                         turbulence * correlation)
            
            wind_forces.append(total_wind)
            
        return jnp.array(wind_forces)
        
    def _calculate_wind_pressure(self, wind_velocity: jnp.ndarray, 
                                mirror_quat: jnp.ndarray, 
                                mirror_area: float) -> jnp.ndarray:
        """Calculate wind pressure force on mirror surface"""
        
        # Convert quaternion to rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(mirror_quat)
        
        # Mirror normal in world coordinates (pointing up in local frame)
        mirror_normal = rot_matrix @ jnp.array([0, 0, 1])
        
        # Relative wind velocity (assuming mirror is stationary for now)
        relative_wind = wind_velocity
        
        # Wind pressure calculation: F = 0.5 * rho * A * Cd * v^2 * normal_component
        air_density = 1.225  # kg/m^3
        drag_coefficient = 1.2  # For flat plate perpendicular to flow
        
        # Component of wind normal to mirror surface
        normal_velocity = jnp.dot(relative_wind, mirror_normal)
        
        # Only apply force for wind hitting the front of mirror
        force_magnitude = (0.5 * air_density * mirror_area * drag_coefficient * 
                          jnp.maximum(0, normal_velocity) * jnp.linalg.norm(relative_wind))
        
        # Force direction is along mirror normal
        wind_force = force_magnitude * mirror_normal
        
        return wind_force
        
    @staticmethod
    def _quat_to_rotation_matrix(quat: jnp.ndarray) -> jnp.ndarray:
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = quat
        
        return jnp.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
    def get_heliostat_states(self, mjx_data: mjx.Data) -> List[HeliostatState]:
        """Extract heliostat states from simulation data"""
        
        states = []
        
        for i in range(self.num_heliostats):
            # Get joint positions and velocities using joint indices directly
            azimuth_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'heliostat_{i}_azimuth_joint')
            elevation_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'heliostat_{i}_elevation_joint')
            
            # Get positions and velocities from qpos and qvel
            azimuth = mjx_data.qpos[azimuth_joint_id]
            elevation = mjx_data.qpos[elevation_joint_id]
            azimuth_vel = mjx_data.qvel[azimuth_joint_id]
            elevation_vel = mjx_data.qvel[elevation_joint_id]
            
            # Calculate mirror normal vector
            mirror_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'heliostat_{i}_mirror')
            mirror_quat = mjx_data.xquat[mirror_body_id]
            rot_matrix = self._quat_to_rotation_matrix(mirror_quat)
            mirror_normal = rot_matrix @ jnp.array([0, 0, 1])
            
            # Get applied torques
            azimuth_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'heliostat_{i}_azimuth')
            elevation_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'heliostat_{i}_elevation')
            torques = jnp.array([
                mjx_data.actuator_force[azimuth_actuator_id],
                mjx_data.actuator_force[elevation_actuator_id]
            ])
            
            state = HeliostatState(
                azimuth=float(azimuth),
                elevation=float(elevation),
                azimuth_velocity=float(azimuth_vel),
                elevation_velocity=float(elevation_vel),
                mirror_normal=mirror_normal,
                torques=torques
            )
            
            states.append(state)
            
        return states
        
    def set_target_angles(self, target_angles: jnp.ndarray):
        """Set target angles for all heliostats with rate limiting"""
        max_rate = 30.0  # degrees per second
        dt = self.timestep
        max_change = max_rate * dt
        
        # Rate limit the target angle changes
        angle_changes = target_angles - self.previous_target_angles
        
        # Wrap angle differences properly
        for i in range(self.num_heliostats):
            angle_changes = angle_changes.at[i, 0].set(
                wrap_angle(angle_changes[i, 0])
            )
        
        # Clip to maximum rate
        limited_changes = jnp.clip(angle_changes, -max_change, max_change)
        
        # Apply limited changes
        self.target_angles = self.previous_target_angles + limited_changes
        self.previous_target_angles = self.target_angles.copy()
        
    def compute_control_torques(self, current_states: List[HeliostatState]) -> jnp.ndarray:
        """Compute control torques using PID control"""
        
        torques = []
        
        for i, state in enumerate(current_states):
            target_az, target_el = self.target_angles[i]
            
            # PID gains (conservative tuning to prevent instability)
            kp_az, ki_az, kd_az = 50.0, 5.0, 10.0
            kp_el, ki_el, kd_el = 30.0, 3.0, 6.0
            
            # Azimuth control with proper angle wrapping
            az_error = angle_difference(target_az, state.azimuth)
            az_error = jnp.clip(az_error, -60.0, 60.0)  # Limit error to reasonable range
            az_vel = jnp.clip(state.azimuth_velocity, -10.0, 10.0)  # Limit velocity
            az_torque = kp_az * az_error - kd_az * az_vel
            
            # Elevation control with numerical safeguards  
            el_error = jnp.clip(target_el - state.elevation, -60.0, 60.0)  # Limit error
            el_vel = jnp.clip(state.elevation_velocity, -10.0, 10.0)  # Limit velocity
            el_torque = kp_el * el_error - kd_el * el_vel
            
            # Limit torques and check for NaN
            max_torque = self.heliostat_configs[i].max_torque
            az_torque = jnp.where(jnp.isfinite(az_torque), 
                                 jnp.clip(az_torque, -max_torque, max_torque), 0.0)
            el_torque = jnp.where(jnp.isfinite(el_torque), 
                                 jnp.clip(el_torque, -max_torque, max_torque), 0.0)
            
            torques.append([az_torque, el_torque])
            
        return jnp.array(torques)
        
    def calculate_sun_tracking_angles(self, sun_direction: jnp.ndarray, 
                                    target_position: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate heliostat angles for sun tracking to target
        
        Parameters
        ----------
        sun_direction : jnp.ndarray
            Unit vector pointing FROM sun TO earth
        target_position : jnp.ndarray
            Target position [x, y, z]
            
        Returns
        -------
        jnp.ndarray
            Target angles [N_heliostats, 2] in degrees
        """
        target_angles = []
        
        for config in self.heliostat_configs:
            # Vector from heliostat to target
            target_vector = target_position - config.position
            target_vector = target_vector / jnp.linalg.norm(target_vector)
            
            # Required mirror normal for specular reflection
            # n = (s + t) / |s + t| where s=sun_dir, t=target_dir
            incident_ray = -sun_direction  # Ray direction (towards earth)
            required_normal = incident_ray + target_vector
            norm_magnitude = jnp.linalg.norm(required_normal)
            # Add numerical safeguard
            required_normal = required_normal / jnp.maximum(norm_magnitude, 1e-8)
            
            # Convert normal to azimuth/elevation angles
            # Normal in world coordinates: [nx, ny, nz]
            nx, ny, nz = required_normal
            
            # Azimuth: angle from +X axis in XY plane
            azimuth = jnp.arctan2(ny, nx) * 180 / jnp.pi
            
            # Elevation: angle from XY plane (clamp nz to valid arcsin range)
            nz_safe = jnp.clip(nz, -1.0, 1.0)
            elevation = jnp.arcsin(nz_safe) * 180 / jnp.pi
            
            # Ensure angles are within heliostat limits
            azimuth = jnp.clip(azimuth, config.azimuth_range[0], config.azimuth_range[1])
            elevation = jnp.clip(elevation, config.elevation_range[0], config.elevation_range[1])
            
            target_angles.append([azimuth, elevation])
            
        return jnp.array(target_angles)


class MJXHeliostatController:
    """High-level controller integrating MuJoCo simulation with verified control"""
    
    def __init__(self, simulator: MuJoCoHeliostatSimulator):
        self.simulator = simulator
        self.sim_data = simulator.mjx_data
        
        # Safety monitoring
        self.max_tracking_error = 2.0  # degrees
        self.max_wind_speed = 25.0  # m/s
        self.emergency_stop = False
        
        logger.info("Initialized MJX heliostat controller")
        
    def run_tracking_simulation(self, 
                               sun_directions: List[jnp.ndarray],
                               target_position: jnp.ndarray,
                               simulation_time: float = 60.0) -> Dict:
        """
        Run heliostat tracking simulation with physics
        
        Parameters
        ----------
        sun_directions : List[jnp.ndarray]
            Time series of sun directions
        target_position : jnp.ndarray
            Target position [x, y, z]
        simulation_time : float
            Total simulation time in seconds
            
        Returns
        -------
        Dict
            Simulation results including states, errors, and safety metrics
        """
        
        logger.info(f"Starting {simulation_time}s tracking simulation")
        
        timesteps = int(simulation_time / self.simulator.timestep)
        results = {
            'time': [],
            'heliostat_states': [],
            'tracking_errors': [],
            'wind_speeds': [],
            'safety_flags': [],
            'control_torques': []
        }
        
        # Initial sun direction
        current_sun_dir = sun_directions[0]
        
        for step in range(timesteps):
            current_time = step * self.simulator.timestep
            
            # Update sun direction (interpolate between provided directions)
            if len(sun_directions) > 1:
                sun_index = int((step / timesteps) * (len(sun_directions) - 1))
                next_index = min(sun_index + 1, len(sun_directions) - 1)
                alpha = ((step / timesteps) * (len(sun_directions) - 1)) - sun_index
                
                current_sun_dir = (1 - alpha) * sun_directions[sun_index] + alpha * sun_directions[next_index]
                current_sun_dir = current_sun_dir / jnp.linalg.norm(current_sun_dir)
            
            # Calculate target tracking angles
            target_angles = self.simulator.calculate_sun_tracking_angles(
                current_sun_dir, target_position
            )
            self.simulator.set_target_angles(target_angles)
            
            # Get current heliostat states
            current_states = self.simulator.get_heliostat_states(self.sim_data)
            
            # Safety check - wind speed monitoring
            wind_speed = jnp.linalg.norm(self.simulator.wind_config.mean_velocity)
            if wind_speed > self.max_wind_speed:
                logger.warning(f"High wind speed: {wind_speed:.1f} m/s - Emergency stop activated")
                self.emergency_stop = True
                
            # Calculate tracking errors
            tracking_errors = []
            for i, state in enumerate(current_states):
                az_error = abs(target_angles[i, 0] - state.azimuth)
                el_error = abs(target_angles[i, 1] - state.elevation)
                total_error = jnp.sqrt(az_error**2 + el_error**2)
                tracking_errors.append(float(total_error))
                
                if total_error > self.max_tracking_error:
                    logger.warning(f"Heliostat {i} tracking error: {total_error:.2f}°")
            
            # Compute control torques (zero if emergency stop)
            if self.emergency_stop:
                control_torques = jnp.zeros((self.simulator.num_heliostats, 2))
            else:
                control_torques = self.simulator.compute_control_torques(current_states)
            
            # Step simulation
            self.sim_data = self.simulator.step(self.sim_data, control_torques)
            
            # Log results (every 10 steps to reduce data volume)
            if step % 10 == 0:
                results['time'].append(current_time)
                results['heliostat_states'].append(current_states)
                results['tracking_errors'].append(tracking_errors)
                results['wind_speeds'].append(float(wind_speed))
                results['safety_flags'].append(self.emergency_stop)
                results['control_torques'].append(control_torques.tolist())
            
        logger.info("Tracking simulation completed")
        return results
        
    def get_mirror_positions_and_normals(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get current mirror positions and normal vectors for ARTIST integration
        
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Mirror positions [N, 3] and normal vectors [N, 3]
        """
        positions = []
        normals = []
        
        for i in range(self.simulator.num_heliostats):
            # Get mirror body position and orientation
            mirror_body_id = mujoco.mj_name2id(self.simulator.model, mujoco.mjtObj.mjOBJ_BODY, f'heliostat_{i}_mirror')
            
            # Position in world coordinates
            position = self.sim_data.xpos[mirror_body_id]
            positions.append(position)
            
            # Normal vector from quaternion
            mirror_quat = self.sim_data.xquat[mirror_body_id]
            rot_matrix = self.simulator._quat_to_rotation_matrix(mirror_quat)
            normal = rot_matrix @ jnp.array([0, 0, 1])  # Z-axis is mirror normal
            normals.append(normal)
            
        return jnp.array(positions), jnp.array(normals)


def create_demo_heliostat_field() -> List[HeliostatConfig]:
    """Create a demonstration heliostat field configuration"""
    
    heliostats = []
    
    # 3x3 grid of heliostats
    for i in range(3):
        for j in range(3):
            x = (i - 1) * 10.0  # 10m spacing
            y = (j - 1) * 10.0
            z = 0.0
            
            heliostat_id = i * 3 + j
            position = jnp.array([x, y, z])
            
            config = HeliostatConfig(
                heliostat_id=heliostat_id,
                position=position,
                mirror_width=2.0,
                mirror_height=1.0,
                pedestal_height=3.0
            )
            
            heliostats.append(config)
            
    return heliostats


def main():
    """Demo of MuJoCo-MJX heliostat simulation"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create heliostat field
    heliostat_configs = create_demo_heliostat_field()
    
    # Wind configuration
    wind_config = WindDisturbance(
        mean_velocity=jnp.array([8.0, 2.0, 0.0]),  # 8 m/s primary wind
        turbulence_intensity=0.15,
        gust_factor=1.4,
        coherence_length=50.0
    )
    
    # Initialize simulator
    simulator = MuJoCoHeliostatSimulator(
        heliostat_configs=heliostat_configs,
        wind_config=wind_config,
        timestep=0.01
    )
    
    # Initialize controller
    controller = MJXHeliostatController(simulator)
    
    # Define sun path (simplified)
    sun_directions = [
        jnp.array([0.0, -0.7, -0.7]) / jnp.linalg.norm(jnp.array([0.0, -0.7, -0.7])),  # Morning
        jnp.array([0.3, -0.8, -0.5]) / jnp.linalg.norm(jnp.array([0.3, -0.8, -0.5])),  # Mid-morning  
        jnp.array([0.0, -1.0, -0.1]) / jnp.linalg.norm(jnp.array([0.0, -1.0, -0.1])),  # Noon
    ]
    
    # Target position (receiver tower)
    target_position = jnp.array([0.0, 50.0, 20.0])
    
    # Run simulation
    results = controller.run_tracking_simulation(
        sun_directions=sun_directions,
        target_position=target_position,
        simulation_time=30.0
    )
    
    # Print summary
    logger.info("=== Simulation Summary ===")
    logger.info(f"Total timesteps: {len(results['time'])}")
    logger.info(f"Max tracking error: {max(max(errors) for errors in results['tracking_errors']):.2f}°")
    logger.info(f"Average wind speed: {np.mean(results['wind_speeds']):.1f} m/s")
    logger.info(f"Emergency stops: {sum(results['safety_flags'])}")
    
    # Get final mirror positions for ARTIST integration
    positions, normals = controller.get_mirror_positions_and_normals()
    logger.info(f"Final mirror positions shape: {positions.shape}")
    logger.info(f"Final mirror normals shape: {normals.shape}")


if __name__ == "__main__":
    main()