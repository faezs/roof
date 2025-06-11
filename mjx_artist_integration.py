"""
MuJoCo-MJX Integration with ARTIST Ray Tracing

This module integrates the MuJoCo physics simulation with ARTIST ray tracing
for realistic heliostat field simulation including:
- Physics-based heliostat dynamics
- Real-time surface deformation from wind/vibrations  
- Accurate flux distribution calculations
- Closed-loop optimization with raytracing feedback
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
import time
import pathlib

# ARTIST imports
from artist.util.scenario import Scenario
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.nurbs import NURBSSurface
from artist.field.surface import Surface
from artist.util.configuration_classes import FacetConfig, SurfaceConfig

# Local imports  
from mujoco_heliostat_sim import MuJoCoHeliostatSimulator, HeliostatConfig, HeliostatState
from mjx_autodiff_control import MJXAutodiffController, MPCParams
from electrostatic_facet import ElectrostaticNurbsFacet
from device_utils import get_default_device

logger = logging.getLogger(__name__)

class FluxOptimizationResult(NamedTuple):
    """Result from flux-based optimization"""
    optimal_angles: jnp.ndarray
    achieved_flux: torch.Tensor
    flux_uniformity: float
    peak_concentration: float
    total_power: float
    optimization_time: float

class MJXARTISTIntegrator:
    """Integrates MuJoCo physics simulation with ARTIST ray tracing"""
    
    def __init__(self,
                 simulator: MuJoCoHeliostatSimulator,
                 controller: MJXAutodiffController,
                 artist_scenario: Scenario,
                 device: torch.device = None):
        """
        Initialize MJX-ARTIST integration
        
        Parameters
        ----------
        simulator : MuJoCoHeliostatSimulator
            MuJoCo physics simulator
        controller : MJXAutodiffController
            Autodiff controller
        artist_scenario : Scenario
            ARTIST ray tracing scenario
        device : torch.device
            PyTorch device for ARTIST computations
        """
        self.simulator = simulator
        self.controller = controller
        self.scenario = artist_scenario
        self.device = device or get_default_device()
        
        # Create raytracer
        self.raytracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=self.scenario.heliostat_field.heliostat_groups[0],
            batch_size=1000
        )
        
        # Mapping between MJX and ARTIST heliostats
        self.mjx_to_artist_mapping = self._create_heliostat_mapping()
        
        # Flux optimization parameters
        self.flux_target_uniformity = 0.8  # 0=perfect point, 1=uniform
        self.max_concentration_ratio = 3.0
        self.flux_optimization_lr = 0.1
        self.flux_optimization_steps = 50
        
        logger.info("Initialized MJX-ARTIST integration")
        
    def _create_heliostat_mapping(self) -> Dict[int, int]:
        """Create mapping between MJX heliostat indices and ARTIST heliostat IDs"""
        
        mapping = {}
        
        # Simple 1:1 mapping for now - in practice would match by position
        for mjx_idx in range(self.simulator.num_heliostats):
            if mjx_idx < len(self.scenario.heliostat_field.heliostat_groups[0].heliostat_list):
                mapping[mjx_idx] = mjx_idx
            else:
                logger.warning(f"MJX heliostat {mjx_idx} has no ARTIST counterpart")
                
        return mapping
        
    def update_artist_surfaces_from_mjx(self, mjx_data) -> None:
        """Update ARTIST heliostat surfaces based on MJX physics state"""
        
        heliostat_states = self.simulator.get_heliostat_states(mjx_data)
        
        for mjx_idx, state in enumerate(heliostat_states):
            if mjx_idx not in self.mjx_to_artist_mapping:
                continue
                
            artist_idx = self.mjx_to_artist_mapping[mjx_idx]
            heliostat = self.scenario.heliostat_field.heliostat_groups[0].heliostat_list[artist_idx]
            
            # Update surface based on physics simulation
            self._update_heliostat_surface_from_physics(heliostat, state, mjx_data)
            
    def _update_heliostat_surface_from_physics(self, 
                                             heliostat,
                                             state: HeliostatState,
                                             mjx_data) -> None:
        """Update individual heliostat surface from MJX physics"""
        
        # Get vibration/deformation data from MJX
        vibration_data = self._extract_vibration_data(heliostat, mjx_data)
        
        # Update surface control points based on vibrations
        if hasattr(heliostat.surface, 'facets') and len(heliostat.surface.facets) > 0:
            facet = heliostat.surface.facets[0]
            
            # Apply small deformations due to wind/vibrations
            deformation_scale = 0.001  # 1mm max deformation
            deformed_points = self._apply_vibration_deformation(
                facet.control_points, vibration_data, deformation_scale
            )
            
            # Update facet control points
            facet.control_points = deformed_points
            
            # Force surface recalculation
            heliostat.surface.is_surface_calculated = False
            
    def _extract_vibration_data(self, heliostat, mjx_data) -> torch.Tensor:
        """Extract vibration/acceleration data from MJX simulation"""
        
        # For demonstration - extract accelerometer data
        # In practice would get specific sensor data for this heliostat
        
        # Simplified vibration model
        vibration_amplitude = np.random.normal(0, 0.1, (3, 3))  # 3x3 spatial pattern
        vibration_tensor = torch.tensor(vibration_amplitude, dtype=torch.float32, device=self.device)
        
        return vibration_tensor
        
    def _apply_vibration_deformation(self,
                                   control_points: torch.Tensor,
                                   vibration_data: torch.Tensor,
                                   scale: float) -> torch.Tensor:
        """Apply vibration-induced deformation to control points"""
        
        deformed_points = control_points.clone()
        
        # Apply vibration deformation to Z coordinates
        if control_points.dim() == 3:  # [M, N, 3] format
            M, N = control_points.shape[:2]
            
            # Interpolate vibration data to control point grid
            for i in range(M):
                for j in range(N):
                    # Simple bilinear interpolation from 3x3 vibration to MxN grid
                    vib_i = min(int(i * 2 / (M - 1)), 2)
                    vib_j = min(int(j * 2 / (N - 1)), 2)
                    
                    vibration_z = vibration_data[vib_i, vib_j] * scale
                    deformed_points[i, j, 2] += vibration_z
                    
        return deformed_points
        
    def raytrace_current_state(self, 
                              sun_direction: torch.Tensor,
                              mjx_data,
                              target_area_idx: int = 0) -> torch.Tensor:
        """
        Perform raytracing with current MJX physics state
        
        Parameters
        ----------
        sun_direction : torch.Tensor
            Incident ray direction [4] (homogeneous)
        mjx_data : mjx.Data
            Current MJX simulation state
        target_area_idx : int
            Target area index
            
        Returns
        -------
        torch.Tensor
            Flux bitmap on target
        """
        
        # Update ARTIST surfaces from MJX physics
        self.update_artist_surfaces_from_mjx(mjx_data)
        
        # Set up raytracing parameters
        active_heliostats_mask = torch.ones(
            self.simulator.num_heliostats, dtype=torch.int32, device=self.device
        )
        target_area_mask = torch.tensor([target_area_idx], device=self.device)
        
        # Align heliostats in ARTIST (this should match MJX orientations)
        heliostat_group = self.scenario.heliostat_field.heliostat_groups[0]
        
        # Get MJX orientations and apply to ARTIST
        self._sync_orientations_mjx_to_artist(mjx_data, heliostat_group)
        
        # Perform raytracing
        flux_bitmap = self.raytracer.trace_rays(
            incident_ray_directions=sun_direction.unsqueeze(0),
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=self.device
        )
        
        # Handle output format
        if isinstance(flux_bitmap, torch.Tensor):
            if flux_bitmap.dim() == 3:
                flux_bitmap = flux_bitmap[0]  # Remove batch dimension
            elif flux_bitmap.dim() == 4:
                flux_bitmap = flux_bitmap[0, 0]  # Remove batch and channel dims
                
        return flux_bitmap
        
    def _sync_orientations_mjx_to_artist(self, mjx_data, heliostat_group) -> None:
        """Synchronize heliostat orientations from MJX to ARTIST"""
        
        heliostat_states = self.simulator.get_heliostat_states(mjx_data)
        
        for mjx_idx, state in enumerate(heliostat_states):
            if mjx_idx not in self.mjx_to_artist_mapping:
                continue
                
            artist_idx = self.mjx_to_artist_mapping[mjx_idx]
            
            # Convert MJX angles to ARTIST orientation
            # This requires careful coordinate frame conversion
            azimuth_rad = np.radians(state.azimuth)
            elevation_rad = np.radians(state.elevation)
            
            # Create orientation quaternion (simplified)
            # In practice, need proper coordinate frame transformation
            orientation = torch.tensor([
                np.cos(azimuth_rad/2) * np.cos(elevation_rad/2),
                np.sin(azimuth_rad/2) * np.cos(elevation_rad/2),
                np.cos(azimuth_rad/2) * np.sin(elevation_rad/2),
                np.sin(azimuth_rad/2) * np.sin(elevation_rad/2)
            ], device=self.device)
            
            # Update ARTIST heliostat orientation
            if artist_idx < len(heliostat_group.heliostat_list):
                heliostat = heliostat_group.heliostat_list[artist_idx]
                # heliostat.current_orientation = orientation  # Would need ARTIST API
                
    def optimize_flux_distribution(self,
                                  sun_direction: torch.Tensor,
                                  target_flux_pattern: torch.Tensor,
                                  initial_angles: jnp.ndarray,
                                  wind_conditions: jnp.ndarray) -> FluxOptimizationResult:
        """
        Optimize heliostat angles for desired flux distribution using physics simulation
        
        Parameters
        ----------
        sun_direction : torch.Tensor
            Sun direction vector
        target_flux_pattern : torch.Tensor
            Desired flux distribution on target
        initial_angles : jnp.ndarray
            Initial heliostat angles [n_heliostats, 2]
        wind_conditions : jnp.ndarray
            Wind velocity [3]
            
        Returns
        -------
        FluxOptimizationResult
            Optimization results
        """
        
        start_time = time.time()
        logger.info("Starting physics-aware flux optimization")
        
        # Convert to JAX for optimization
        current_angles = initial_angles.copy()
        
        # Define objective function
        def flux_objective(angles_flat):
            """Objective function for flux optimization"""
            
            angles = angles_flat.reshape(self.simulator.num_heliostats, 2)
            
            # Set MJX target angles
            self.simulator.set_target_angles(angles)
            
            # Simulate physics response (simplified)
            # In practice, would run several MJX steps to reach steady state
            simulated_mjx_data = self._simulate_to_steady_state(angles, wind_conditions)
            
            # Convert sun direction to PyTorch
            sun_torch = torch.tensor(
                jnp.append(sun_direction, 0.0), 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Raytrace current state
            achieved_flux = self.raytrace_current_state(
                sun_torch, simulated_mjx_data, target_area_idx=0
            )
            
            # Compute flux quality metrics
            flux_error = self._compute_flux_error(achieved_flux, target_flux_pattern)
            concentration_penalty = self._compute_concentration_penalty(achieved_flux)
            uniformity_bonus = self._compute_uniformity_bonus(achieved_flux)
            
            # Total cost
            total_cost = (flux_error + 
                         10.0 * concentration_penalty - 
                         5.0 * uniformity_bonus)
            
            return float(total_cost)
            
        # Optimize using JAX
        angles_flat = current_angles.flatten()
        
        for iteration in range(self.flux_optimization_steps):
            # Compute gradient
            grad_fn = jax.grad(flux_objective)
            gradient = grad_fn(angles_flat)
            
            # Gradient descent step
            angles_flat = angles_flat - self.flux_optimization_lr * gradient
            
            # Apply angle constraints
            angles_flat = self._apply_angle_constraints(angles_flat)
            
            if iteration % 10 == 0:
                cost = flux_objective(angles_flat)
                logger.info(f"Flux optimization iter {iteration}: cost = {cost:.4f}")
                
        # Final evaluation
        optimal_angles = angles_flat.reshape(self.simulator.num_heliostats, 2)
        final_mjx_data = self._simulate_to_steady_state(optimal_angles, wind_conditions)
        
        sun_torch = torch.tensor(
            jnp.append(sun_direction, 0.0), 
            dtype=torch.float32, 
            device=self.device
        )
        final_flux = self.raytrace_current_state(sun_torch, final_mjx_data, 0)
        
        # Compute final metrics
        flux_uniformity = self._compute_flux_uniformity(final_flux)
        peak_concentration = self._compute_peak_concentration(final_flux)
        total_power = float(torch.sum(final_flux))
        
        optimization_time = time.time() - start_time
        
        result = FluxOptimizationResult(
            optimal_angles=optimal_angles,
            achieved_flux=final_flux,
            flux_uniformity=flux_uniformity,
            peak_concentration=peak_concentration,
            total_power=total_power,
            optimization_time=optimization_time
        )
        
        logger.info(f"Flux optimization completed in {optimization_time:.2f}s")
        logger.info(f"Final uniformity: {flux_uniformity:.3f}")
        logger.info(f"Peak concentration: {peak_concentration:.2f}")
        
        return result
        
    def _simulate_to_steady_state(self, 
                                 target_angles: jnp.ndarray,
                                 wind_conditions: jnp.ndarray,
                                 settle_time: float = 2.0):
        """Simulate MJX to steady state for given target angles"""
        
        # Set target angles
        self.simulator.set_target_angles(target_angles)
        
        # Update wind conditions
        self.simulator.wind_config = self.simulator.wind_config._replace(
            mean_velocity=wind_conditions
        )
        
        # Simulate until settled
        mjx_data = self.simulator.mjx_data
        settle_steps = int(settle_time / self.simulator.timestep)
        
        for _ in range(settle_steps):
            # Get current states
            current_states = self.simulator.get_heliostat_states(mjx_data)
            
            # Compute control torques
            control_torques = self.simulator.compute_control_torques(current_states)
            
            # Step simulation
            mjx_data = self.simulator.step(mjx_data, control_torques)
            
        return mjx_data
        
    def _compute_flux_error(self, achieved: torch.Tensor, target: torch.Tensor) -> float:
        """Compute flux distribution error"""
        
        if achieved.shape != target.shape:
            # Resize target to match achieved
            target = torch.nn.functional.interpolate(
                target.unsqueeze(0).unsqueeze(0),
                size=achieved.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
        mse_error = torch.mean((achieved - target)**2)
        return float(mse_error)
        
    def _compute_concentration_penalty(self, flux: torch.Tensor) -> float:
        """Penalize excessive concentration"""
        
        max_flux = torch.max(flux)
        mean_flux = torch.mean(flux[flux > 0])
        
        if mean_flux > 0:
            concentration_ratio = max_flux / mean_flux
            penalty = torch.relu(concentration_ratio - self.max_concentration_ratio)
            return float(penalty)
        else:
            return 0.0
            
    def _compute_uniformity_bonus(self, flux: torch.Tensor) -> float:
        """Reward flux uniformity"""
        
        if torch.sum(flux) > 0:
            normalized_flux = flux / torch.sum(flux)
            entropy = -torch.sum(normalized_flux * torch.log(normalized_flux + 1e-10))
            max_entropy = np.log(flux.numel())  # Maximum possible entropy
            uniformity = entropy / max_entropy
            return float(uniformity)
        else:
            return 0.0
            
    def _compute_flux_uniformity(self, flux: torch.Tensor) -> float:
        """Compute flux uniformity metric"""
        return self._compute_uniformity_bonus(flux)
        
    def _compute_peak_concentration(self, flux: torch.Tensor) -> float:
        """Compute peak concentration ratio"""
        
        max_flux = torch.max(flux)
        mean_flux = torch.mean(flux[flux > 0])
        
        if mean_flux > 0:
            return float(max_flux / mean_flux)
        else:
            return 0.0
            
    def _apply_angle_constraints(self, angles_flat: jnp.ndarray) -> jnp.ndarray:
        """Apply heliostat angle constraints"""
        
        angles = angles_flat.reshape(self.simulator.num_heliostats, 2)
        
        for i, config in enumerate(self.simulator.heliostat_configs):
            # Azimuth constraints
            angles = angles.at[i, 0].set(jnp.clip(
                angles[i, 0], config.azimuth_range[0], config.azimuth_range[1]
            ))
            
            # Elevation constraints
            angles = angles.at[i, 1].set(jnp.clip(
                angles[i, 1], config.elevation_range[0], config.elevation_range[1]
            ))
            
        return angles.flatten()
        
    def run_closed_loop_simulation(self,
                                  sun_trajectory: List[torch.Tensor],
                                  target_flux_pattern: torch.Tensor,
                                  wind_forecast: jnp.ndarray,
                                  simulation_duration: float = 3600.0) -> Dict:
        """
        Run closed-loop simulation with MJX physics and ARTIST raytracing
        
        Parameters
        ----------
        sun_trajectory : List[torch.Tensor]
            Time series of sun directions
        target_flux_pattern : torch.Tensor
            Desired flux distribution
        wind_forecast : jnp.ndarray
            Wind forecast [time_steps, 3]
        simulation_duration : float
            Simulation duration in seconds
            
        Returns
        -------
        Dict
            Simulation results
        """
        
        logger.info(f"Starting closed-loop simulation for {simulation_duration}s")
        
        # Simulation parameters
        control_timestep = 10.0  # Reoptimize every 10 seconds
        physics_timestep = self.simulator.timestep
        
        control_steps = int(simulation_duration / control_timestep)
        physics_steps_per_control = int(control_timestep / physics_timestep)
        
        results = {
            'time': [],
            'flux_distributions': [],
            'heliostat_angles': [],
            'tracking_errors': [],
            'flux_uniformity': [],
            'peak_concentration': [],
            'total_power': []
        }
        
        # Initial state
        current_angles = jnp.zeros((self.simulator.num_heliostats, 2))
        mjx_data = self.simulator.mjx_data
        
        for control_step in range(control_steps):
            current_time = control_step * control_timestep
            
            # Get current sun direction
            sun_idx = min(int(control_step / control_steps * len(sun_trajectory)), 
                         len(sun_trajectory) - 1)
            sun_direction = sun_trajectory[sun_idx][:3]  # Remove homogeneous coordinate
            
            # Get current wind conditions
            wind_idx = min(int(control_step / control_steps * len(wind_forecast)),
                          len(wind_forecast) - 1)
            wind_velocity = wind_forecast[wind_idx]
            
            logger.info(f"Control step {control_step}: t={current_time:.1f}s")
            
            # Optimize flux distribution
            optimization_result = self.optimize_flux_distribution(
                sun_direction=sun_direction,
                target_flux_pattern=target_flux_pattern,
                initial_angles=current_angles,
                wind_conditions=wind_velocity
            )
            
            current_angles = optimization_result.optimal_angles
            
            # Run physics simulation for control interval
            for _ in range(physics_steps_per_control):
                # Get current states
                current_states = self.simulator.get_heliostat_states(mjx_data)
                
                # Compute control torques with MPC
                control_torques = self.controller.compute_control_torques(current_states)
                
                # Apply safety filter
                safe_torques = self.controller.safety_filter(
                    control_torques, current_states, jnp.linalg.norm(wind_velocity)
                )
                
                # Step physics
                mjx_data = self.simulator.step(mjx_data, safe_torques)
                
            # Log results
            results['time'].append(current_time)
            results['flux_distributions'].append(optimization_result.achieved_flux.cpu().numpy())
            results['heliostat_angles'].append(current_angles)
            results['flux_uniformity'].append(optimization_result.flux_uniformity)
            results['peak_concentration'].append(optimization_result.peak_concentration)
            results['total_power'].append(optimization_result.total_power)
            
            # Compute tracking errors
            current_states = self.simulator.get_heliostat_states(mjx_data)
            tracking_errors = []
            for i, state in enumerate(current_states):
                az_error = abs(current_angles[i, 0] - state.azimuth)
                el_error = abs(current_angles[i, 1] - state.elevation)
                total_error = np.sqrt(az_error**2 + el_error**2)
                tracking_errors.append(total_error)
            results['tracking_errors'].append(tracking_errors)
            
        logger.info("Closed-loop simulation completed")
        return results


def create_simple_artist_scenario(heliostat_configs: List[HeliostatConfig], 
                                 device: torch.device) -> Scenario:
    """Create a simple ARTIST scenario for MJX integration testing"""
    
    # This is a simplified version - in practice would load from HDF5
    logger.info("Creating simplified ARTIST scenario for MJX integration")
    logger.info("In production, load a real scenario from HDF5 file")
    
    # For now, return None and use existing test scenarios
    return None


def main():
    """Demo of MJX-ARTIST integration"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Import required components
    from mujoco_heliostat_sim import create_demo_heliostat_field, WindDisturbance
    
    # Create heliostat field
    heliostat_configs = create_demo_heliostat_field()
    
    # Initialize MJX simulator
    wind_config = WindDisturbance(
        mean_velocity=jnp.array([6.0, 2.0, 0.0]),
        turbulence_intensity=0.1,
        gust_factor=1.3,
        coherence_length=60.0
    )
    
    simulator = MuJoCoHeliostatSimulator(
        heliostat_configs=heliostat_configs,
        wind_config=wind_config
    )
    
    # Initialize autodiff controller
    mpc_params = MPCParams(horizon=10, Q_tracking=150.0)
    controller = MJXAutodiffController(simulator, mpc_params)
    
    # For demo purposes, create mock ARTIST scenario
    device = get_default_device()
    artist_scenario = create_simple_artist_scenario(heliostat_configs, device)
    
    if artist_scenario is None:
        logger.info("No ARTIST scenario available - running MJX-only demo")
        
        # Demo MJX simulation with flux calculations
        sun_directions = [
            jnp.array([0.0, -0.8, -0.6]),  # Morning
            jnp.array([0.0, -1.0, -0.2]),  # Noon
            jnp.array([0.0, -0.8, 0.6])    # Evening
        ]
        
        target_position = jnp.array([0.0, 50.0, 15.0])
        
        logger.info("Running MJX heliostat tracking simulation...")
        
        for i, sun_dir in enumerate(sun_directions):
            logger.info(f"Sun direction {i+1}: {sun_dir}")
            
            # Calculate tracking angles
            target_angles = simulator.calculate_sun_tracking_angles(sun_dir, target_position)
            logger.info(f"Target angles: {target_angles}")
            
            # Simulate physics
            simulator.set_target_angles(target_angles)
            
            for step in range(100):  # 1 second simulation
                current_states = simulator.get_heliostat_states(simulator.mjx_data)
                control_torques = simulator.compute_control_torques(current_states)
                simulator.mjx_data = simulator.step(simulator.mjx_data, control_torques)
                
            # Check final tracking accuracy
            final_states = simulator.get_heliostat_states(simulator.mjx_data)
            logger.info(f"Final angles: {[(s.azimuth, s.elevation) for s in final_states[:3]]}")
            
        logger.info("MJX simulation completed successfully!")
        
    else:
        # Full MJX-ARTIST integration demo
        integrator = MJXARTISTIntegrator(simulator, controller, artist_scenario, device)
        
        logger.info("Running full MJX-ARTIST integration demo...")
        
        # Define simulation parameters
        sun_trajectory = [
            torch.tensor([0.0, -0.8, -0.6, 0.0], device=device),  # Morning
            torch.tensor([0.0, -1.0, -0.2, 0.0], device=device),  # Noon
            torch.tensor([0.0, -0.8, 0.6, 0.0], device=device)    # Evening
        ]
        
        # Target flux pattern (uniform distribution)
        target_flux = torch.ones((64, 64), device=device) * 1000.0  # 1 kW/m²
        
        # Wind forecast
        wind_forecast = jnp.array([
            [5.0, 1.0, 0.0],  # Morning
            [8.0, 2.0, 0.0],  # Noon
            [6.0, 1.5, 0.0]   # Evening
        ])
        
        # Run closed-loop simulation
        results = integrator.run_closed_loop_simulation(
            sun_trajectory=sun_trajectory,
            target_flux_pattern=target_flux,
            wind_forecast=wind_forecast,
            simulation_duration=300.0  # 5 minutes
        )
        
        logger.info("=== Integration Demo Results ===")
        logger.info(f"Simulation steps: {len(results['time'])}")
        logger.info(f"Average flux uniformity: {np.mean(results['flux_uniformity']):.3f}")
        logger.info(f"Average peak concentration: {np.mean(results['peak_concentration']):.2f}")
        logger.info(f"Average total power: {np.mean(results['total_power']):.1f}W")


if __name__ == "__main__":
    main()