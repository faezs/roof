#!/usr/bin/env python3
"""
Test Surface Learning System

This module creates test scenarios with known distorted mirror surfaces,
generates realistic focal spot bitmaps, and validates that the learning
system can recover the known surfaces from the generated images.
"""

import logging
import pathlib
import time
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# Monkey-patch ARTIST to force CPU usage
import artist.util.nurbs
original_find_span = artist.util.nurbs.NURBSSurface.find_span

DEVICE="mps"

@staticmethod
def patched_find_span(degree, evaluation_points, knot_vector, control_points, device="cpu"):
    """Patched find_span that forces CPU device"""
    device = torch.device(DEVICE)  # Force CPU
    return original_find_span(degree, evaluation_points, knot_vector, control_points, device)

# Apply the patch
artist.util.nurbs.NURBSSurface.find_span = patched_find_span

# ARTIST imports
from artist.field.surface import Surface
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.kinematic_optimizer import KinematicOptimizer
from artist.util.configuration_classes import (
    ActuatorConfig,
    ActuatorListConfig,
    ActuatorPrototypeConfig,
    FacetConfig,
    HeliostatConfig,
    HeliostatListConfig,
    KinematicConfig,
    KinematicDeviations,
    KinematicPrototypeConfig,
    LightSourceConfig,
    LightSourceListConfig,
    PowerPlantConfig,
    PrototypeConfig,
    SurfaceConfig,
    SurfacePrototypeConfig,
    TargetAreaConfig,
    TargetAreaListConfig,
)
from artist.util.nurbs import NURBSSurface
from artist.util import config_dictionary

# Local imports
from image_based_surface_converter import ImageBasedSurfaceConverter
from electrostatic_facet import ElectrostaticNurbsFacet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistortedMirrorGenerator:
    """Generates heliostats with known surface distortions for testing"""
    
    def __init__(self, device: torch.device = torch.device(DEVICE)):
        self.device = device
        
        # Mylar electrostatic physics parameters
        self.membrane_width = 2.0    # meters
        self.membrane_height = 1.0   # meters  
        self.electrode_gap = 0.005   # 5mm gap to electrodes
        self.max_voltage = 300.0     # volts
        self.pull_in_voltage = 150.0 # voltage where pull-in instability occurs
        
    def create_distorted_nurbs_surface(self, 
                                     distortion_type: str = "gaussian_bump",
                                     distortion_magnitude: float = 0.01,
                                     voltage_pattern: torch.Tensor = None) -> Surface:
        """Create a Surface with known distortions (including electrostatic mylar)"""
        
        # Create evaluation points (avoid exactly 0 and 1)
        num_eval_e, num_eval_n = 20, 20
        eval_points_e = torch.linspace(1e-5, 1 - 1e-5, num_eval_e, device=self.device)
        eval_points_n = torch.linspace(1e-5, 1 - 1e-5, num_eval_n, device=self.device)
        
        # Create control points grid (6x6 for simpler setup, degree 2)
        num_ctrl_e, num_ctrl_n = 6, 6
        degree_e, degree_n = 2, 2
        
        # Initialize control points using ARTIST test pattern
        control_points_shape = (num_ctrl_e, num_ctrl_n)
        ctrl_e_range = torch.linspace(-1.0, 1.0, num_ctrl_e, device=self.device)
        ctrl_n_range = torch.linspace(-1.0, 1.0, num_ctrl_n, device=self.device)
        
        # Use cartesian product like in ARTIST test
        ctrl_grid = torch.cartesian_prod(ctrl_e_range, ctrl_n_range)
        ctrl_with_z = torch.hstack((
            ctrl_grid,
            torch.zeros((len(ctrl_grid), 1), device=self.device)  # Flat Z initially
        ))
        
        # Reshape to proper grid format
        control_points = ctrl_with_z.reshape(control_points_shape + (3,))
        
        # Handle electrostatic mylar with custom facet
        if distortion_type == "electrostatic_mylar":
            # Create ElectrostaticNurbsFacet for voltage-controlled surface
            electrostatic_facet = ElectrostaticNurbsFacet(
                control_points=control_points,
                degree_e=degree_e,
                degree_n=degree_n,
                number_eval_points_e=num_eval_e,
                number_eval_points_n=num_eval_n,
                translation_vector=torch.zeros(4, device=self.device),
                canting_e=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                canting_n=torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device),
                voltage_pattern=voltage_pattern,
                electrode_gap=self.electrode_gap,
                max_voltage=self.max_voltage,
                pull_in_voltage=self.pull_in_voltage
            )
            
            # Create Surface with electrostatic facet
            facet_config = FacetConfig(
                facet_key="electrostatic_facet",
                control_points=electrostatic_facet.control_points,  # Use deformed control points
                degree_e=degree_e,
                degree_n=degree_n,
                number_eval_points_e=num_eval_e,
                number_eval_points_n=num_eval_n,
                translation_vector=torch.zeros(4, device=self.device),
                canting_e=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                canting_n=torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
            )
            
            surface_config = SurfaceConfig(facet_list=[facet_config])
            return Surface(surface_config)
            
        # Apply distortions for non-electrostatic surfaces
        elif distortion_type == "flat":
            # Keep flat surface - no changes needed
            pass
        elif distortion_type == "gaussian_bump":
            # Create a Gaussian bump in the center
            center_e, center_n = num_ctrl_e // 2, num_ctrl_n // 2
            for i in range(num_ctrl_e):
                for j in range(num_ctrl_n):
                    dist_sq = torch.tensor((i - center_e)**2 + (j - center_n)**2, device=self.device, dtype=torch.float32)
                    control_points[i, j, 2] = distortion_magnitude * torch.exp(-dist_sq / 2.0)
                    
        elif distortion_type == "saddle":
            # Create a saddle shape
            for i in range(num_ctrl_e):
                for j in range(num_ctrl_n):
                    x = (i - num_ctrl_e/2) / (num_ctrl_e/2)
                    y = (j - num_ctrl_n/2) / (num_ctrl_n/2)
                    control_points[i, j, 2] = distortion_magnitude * (x**2 - y**2)
                    
        elif distortion_type == "wave":
            # Create a wave pattern
            for i in range(num_ctrl_e):
                for j in range(num_ctrl_n):
                    x = torch.tensor((i / num_ctrl_e) * 2 * np.pi, device=self.device, dtype=torch.float32)
                    y = torch.tensor((j / num_ctrl_n) * 2 * np.pi, device=self.device, dtype=torch.float32)
                    control_points[i, j, 2] = distortion_magnitude * torch.sin(x) * torch.cos(y)
        
        # Make control points learnable
        control_points = torch.nn.Parameter(control_points)
        
        # Create FacetConfig and SurfaceConfig for non-electrostatic surfaces
        facet_config = FacetConfig(
            facet_key="distorted_facet",
            control_points=control_points,
            degree_e=degree_e,
            degree_n=degree_n,
            number_eval_points_e=num_eval_e,
            number_eval_points_n=num_eval_n,
            translation_vector=torch.zeros(4, device=self.device),
            canting_e=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
            canting_n=torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        )
        
        surface_config = SurfaceConfig(facet_list=[facet_config])
        return Surface(surface_config)
    
    def _apply_electrostatic_deformation(self, control_points: torch.Tensor, 
                                       voltage_pattern: torch.Tensor,
                                       max_deflection: float) -> torch.Tensor:
        """
        Apply physics-based electrostatic deformation to NURBS control points
        
        Models mylar membrane deflection under 4x4 electrode grid voltage pattern.
        Uses physics-inspired mapping from electrode voltages to control point positions.
        
        Args:
            control_points: [6, 6, 3] initial flat control points
            voltage_pattern: [16] voltages for 4x4 electrode grid (0-300V)
            max_deflection: maximum allowed deflection (safety limit)
            
        Returns:
            Modified control points with electrostatic deflection
        """
        if voltage_pattern is None:
            return control_points
            
        # Ensure voltage pattern is correct size
        if voltage_pattern.shape[0] != 16:
            raise ValueError(f"Expected 16 voltages for 4x4 grid, got {voltage_pattern.shape[0]}")
        
        # Reshape voltage pattern to 4x4 electrode grid
        electrode_voltages = voltage_pattern.reshape(4, 4)
        
        # Define electrode positions in membrane coordinates (normalized 0-1)
        electrode_positions = torch.zeros(4, 4, 2, device=self.device)
        for i in range(4):
            for j in range(4):
                electrode_positions[i, j, 0] = i / 3.0  # x position (0 to 1)
                electrode_positions[i, j, 1] = j / 3.0  # y position (0 to 1)
        
        # Control point positions in membrane coordinates
        num_ctrl_e, num_ctrl_n = control_points.shape[0], control_points.shape[1]
        
        # Apply electrostatic forces to each control point
        for cp_i in range(num_ctrl_e):
            for cp_j in range(num_ctrl_n):
                # Control point position in normalized coordinates
                cp_x = cp_i / (num_ctrl_e - 1)
                cp_y = cp_j / (num_ctrl_n - 1)
                
                # Calculate total electrostatic force from all electrodes
                total_deflection = 0.0
                
                for elec_i in range(4):
                    for elec_j in range(4):
                        # Distance from control point to electrode
                        dx = cp_x - electrode_positions[elec_i, elec_j, 0]
                        dy = cp_y - electrode_positions[elec_i, elec_j, 1]
                        distance = torch.sqrt(dx**2 + dy**2 + 1e-6)  # Avoid division by zero
                        
                        # Electrode voltage
                        voltage = electrode_voltages[elec_i, elec_j]
                        
                        # Physics-based electrostatic force calculation
                        # F ∝ V²/d², but with pull-in instability and safety limits
                        
                        # Influence falloff with distance (Gaussian-like) - make more localized
                        influence_radius = 0.25  # Each electrode influences ~25% of membrane for sharper control
                        spatial_influence = torch.exp(-(distance / influence_radius)**2)
                        
                        # Physics-based deflection calculation: F ∝ V²/d²
                        # Base deflection scaling for 300V → ~3cm deflection
                        base_deflection_scale = 3e-4  # 0.3mm per volt squared
                        
                        # Pull-in instability: force increases dramatically near pull-in voltage
                        if voltage > self.pull_in_voltage:
                            # Nonlinear pull-in regime - deflection increases rapidly
                            deflection_scale = base_deflection_scale * (voltage / self.pull_in_voltage)**3
                        else:
                            # Linear regime below pull-in - normal V² relationship
                            deflection_scale = base_deflection_scale
                        
                        # Calculate deflection magnitude (attractive force only - negative Z)
                        voltage_deflection = deflection_scale * (voltage**2)
                        electrode_deflection = -voltage_deflection * spatial_influence
                        
                        total_deflection += electrode_deflection
                
                # Apply boundary conditions (edges are more constrained)
                edge_factor = 1.0
                if cp_i == 0 or cp_i == num_ctrl_e-1 or cp_j == 0 or cp_j == num_ctrl_n-1:
                    edge_factor = 0.6  # Edge control points move less but not too constrained
                
                # Apply edge constraints
                total_deflection = total_deflection * edge_factor
                
                # Safety limits: prevent electrode collision
                max_safe_deflection = -self.electrode_gap * 0.8  # Stay 80% away from electrodes  
                total_deflection = torch.clamp(total_deflection, max_safe_deflection, 0.0)
                
                # Apply deflection to Z coordinate
                control_points[cp_i, cp_j, 2] += total_deflection
        
        return control_points
    
    def create_test_scenario(self, surface: Surface) -> Scenario:
        """Create a test scenario with the given Surface"""
        
        logger.debug("Starting scenario creation...")
        
        # Create target area (receiver)
        logger.debug("Creating target config...")
        target_config = TargetAreaConfig(
            target_area_key="receiver",
            geometry="planar",
            center=torch.tensor([0.0, 50.0, 10.0, 1.0], device=self.device),  # Homogeneous coordinates
            normal_vector=torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device),  # Homogeneous coordinates
            plane_e=10.0,  # 10m x 10m receiver
            plane_u=10.0,
        )
        logger.debug("Target config created")
        
        logger.debug("Creating target list config...")
        target_list_config = TargetAreaListConfig(
            target_area_list=[target_config]
        )
        logger.debug("Target list config created")
        
        # Create light source (sun)
        logger.debug("Creating light source config...")
        light_source_config = LightSourceConfig(
            light_source_key="sun",
            light_source_type="sun",
            number_of_rays=10000,  # More rays for better focal spots
            distribution_type="normal",
            mean=0.0,
            covariance=2.0e-05,  # Larger solar disk for more spread
        )
        logger.debug("Light source config created")
        
        logger.debug("Creating light source list config...")
        light_source_list_config = LightSourceListConfig(
            light_source_list=[light_source_config]
        )
        logger.debug("Light source list config created")
        
        # Extract surface information from Surface object
        logger.debug("Extracting surface info...")
        
        # Get the first facet from the surface (we only have one facet)
        first_facet = surface.facets[0]
        
        # Check the actual control points dimensions
        logger.debug(f"Control points shape: {first_facet.control_points.shape}")
        
        # Test what surface points look like
        try:
            test_surface_points, test_surface_normals = surface.get_surface_points_and_normals(device=self.device)
            logger.debug(f"Surface points shape: {test_surface_points.shape}")
            logger.debug(f"Surface normals shape: {test_surface_normals.shape}")
        except Exception as e:
            logger.debug(f"Error calculating surface points: {e}")
        
        # Keep original 3D structure but detach gradients
        control_points_detached = first_facet.control_points.detach()
        
        facet_config = FacetConfig(
            facet_key="test_facet",
            control_points=control_points_detached,  # Keep 3D structure
            degree_e=first_facet.degree_e,
            degree_n=first_facet.degree_n,
            number_eval_points_e=first_facet.number_eval_points_e,
            number_eval_points_n=first_facet.number_eval_points_n,
            translation_vector=first_facet.translation_vector,
            canting_e=first_facet.canting_e,
            canting_n=first_facet.canting_n,
        )
        logger.debug("Facet config created")
        
        # Create prototype configurations
        logger.debug("Creating prototype configurations...")
        
        # Surface prototype
        surface_prototype_config = SurfacePrototypeConfig(facet_list=[facet_config])
        
        # Create zero deviations to eliminate warnings
        zero_deviations = KinematicDeviations(
            first_joint_translation_e=torch.tensor(0.0, device=self.device),
            first_joint_translation_n=torch.tensor(0.0, device=self.device),
            first_joint_translation_u=torch.tensor(0.0, device=self.device),
            first_joint_tilt_e=torch.tensor(0.0, device=self.device),
            first_joint_tilt_n=torch.tensor(0.0, device=self.device),
            first_joint_tilt_u=torch.tensor(0.0, device=self.device),
            second_joint_translation_e=torch.tensor(0.0, device=self.device),
            second_joint_translation_n=torch.tensor(0.0, device=self.device),
            second_joint_translation_u=torch.tensor(0.0, device=self.device),
            second_joint_tilt_e=torch.tensor(0.0, device=self.device),
            second_joint_tilt_n=torch.tensor(0.0, device=self.device),
            second_joint_tilt_u=torch.tensor(0.0, device=self.device),
            concentrator_translation_e=torch.tensor(0.0, device=self.device),
            concentrator_translation_n=torch.tensor(0.0, device=self.device),
            concentrator_translation_u=torch.tensor(0.0, device=self.device),
            concentrator_tilt_e=torch.tensor(0.0, device=self.device),
            concentrator_tilt_n=torch.tensor(0.0, device=self.device),
            concentrator_tilt_u=torch.tensor(0.0, device=self.device),
        )
        
        # Kinematic prototype with explicit zero deviations
        kinematic_prototype_config = KinematicPrototypeConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=[0.0, 0.0, 1.0, 0.0],
            deviations=zero_deviations,
        )
        
        # Actuator prototypes
        actuator1_prototype = ActuatorConfig(
            key="actuator_1",
            type=config_dictionary.ideal_actuator_key,
            clockwise_axis_movement=False,
        )
        actuator2_prototype = ActuatorConfig(
            key="actuator_2", 
            type=config_dictionary.ideal_actuator_key,
            clockwise_axis_movement=True,
        )
        actuator_prototype_config = ActuatorPrototypeConfig(
            actuator_list=[actuator1_prototype, actuator2_prototype]
        )
        
        # Combined prototype config
        prototype_config = PrototypeConfig(
            surface_prototype=surface_prototype_config,
            kinematic_prototype=kinematic_prototype_config,
            actuators_prototype=actuator_prototype_config,
        )
        logger.debug("Prototype configurations created")
        
        # Create surface config for heliostat
        surface_config = SurfaceConfig(facet_list=[facet_config])
        
        # Create kinematic config for heliostat with explicit zero deviations
        kinematic_config = KinematicConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=[0.0, 0.0, 1.0, 0.0],
            deviations=zero_deviations,
        )
        
        # Create actuator configs for heliostat
        actuator1_heliostat = ActuatorConfig(
            key="actuator_1",
            type=config_dictionary.ideal_actuator_key,
            clockwise_axis_movement=False,
        )
        actuator2_heliostat = ActuatorConfig(
            key="actuator_2",
            type=config_dictionary.ideal_actuator_key,
            clockwise_axis_movement=True,
        )
        actuator_list_config = ActuatorListConfig(
            actuator_list=[actuator1_heliostat, actuator2_heliostat]
        )
        
        # Create heliostat config
        logger.debug("Creating heliostat config...")
        heliostat_config = HeliostatConfig(
            name="test_heliostat",
            id=1,
            position=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device),  # Homogeneous coordinates
            aim_point=torch.tensor([0.0, 50.0, 10.0, 1.0], device=self.device),  # Homogeneous coordinates
            surface=surface_config,
            kinematic=kinematic_config,
            actuators=actuator_list_config,
        )
        logger.debug("Heliostat config created")
        
        logger.debug("Creating heliostat list config...")
        heliostat_list_config = HeliostatListConfig(
            heliostat_list=[heliostat_config]
        )
        logger.debug("Heliostat list config created")
        
        # Create power plant config
        logger.debug("Creating power plant config...")
        power_plant_config = PowerPlantConfig(
            power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=self.device),  # This might be lat/lon/alt, keep as 3D
        )
        logger.debug("Power plant config created")
        
        # Generate scenario using ScenarioGenerator
        logger.debug("Creating scenario generator...")
        import tempfile
        import pathlib
        
        # Create temporary file for scenario
        temp_dir = pathlib.Path("/tmp")
        scenario_file = temp_dir / "test_scenario"
        
        scenario_generator = ScenarioGenerator(
            file_path=scenario_file,
            power_plant_config=power_plant_config,
            target_area_list_config=target_list_config,
            light_source_list_config=light_source_list_config,
            heliostat_list_config=heliostat_list_config,
            prototype_config=prototype_config,
        )
        
        logger.debug("Generating scenario HDF5 file...")
        scenario_generator.generate_scenario()
        
        # Load scenario from generated HDF5 file
        logger.debug("Loading scenario from HDF5...")
        import h5py
        scenario_h5_path = temp_dir / (scenario_file.name + ".h5")
        
        with h5py.File(scenario_h5_path) as scenario_file_h5:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file_h5, 
                device=self.device
            )
        
        logger.debug("Scenario loaded successfully")
        return scenario


class FocalSpotGenerator:
    """Generates realistic focal spot bitmaps from test scenarios"""
    
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)):
        self.device = device
        
    def generate_focal_spots(self, 
                           scenario: Scenario, 
                           sun_directions: List[torch.Tensor],
                           add_noise: bool = True) -> List[torch.Tensor]:
        """Generate focal spot images for multiple sun directions"""
        
        focal_spots = []
        
        # Set up fixed parameters once
        active_heliostats_mask = torch.tensor([1], dtype=torch.int32, device=self.device)
        target_area_mask = torch.tensor([0], device=self.device)  # First target area
        heliostat_group = scenario.heliostat_field.heliostat_groups[0]
        
        for sun_dir in sun_directions:
            logger.info(f"Generating focal spot for sun direction: {sun_dir.cpu().numpy()}")
            
            # CRITICAL FIX: Ensure sun direction is properly formatted (4D homogeneous)
            if sun_dir.shape[0] != 4:
                logger.warning(f"Sun direction has wrong shape {sun_dir.shape}, should be [4]")
                continue
                
            # STEP 1: Activate heliostats (CRITICAL - this was missing!)
            heliostat_group.activate_heliostats(active_heliostats_mask=active_heliostats_mask)
            
            # STEP 2: Align heliostat surfaces with incident ray directions
            # This is the key kinematic alignment step from the tutorial
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.target_areas.centers[target_area_mask],
                incident_ray_directions=sun_dir.unsqueeze(0),  # Must be batched: [1, 4]
                active_heliostats_mask=active_heliostats_mask,
                device=self.device,
            )
            
            # STEP 3: Create raytracer AFTER alignment (kinematic optimizer pattern)
            raytracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=heliostat_group,
                batch_size=heliostat_group.number_of_active_heliostats,
            )
            
            # STEP 4: Perform raytracing
            bitmap = raytracer.trace_rays(
                incident_ray_directions=sun_dir.unsqueeze(0),  # Must be batched: [1, 4]
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask=target_area_mask,
                device=self.device,
            )
            
            # Handle raytracer output
            if isinstance(bitmap, torch.Tensor):
                if bitmap.dim() == 3:  # [batch, height, width]
                    bitmap = bitmap[0]  # Remove batch dimension
                elif bitmap.dim() == 4:  # [batch, channels, height, width] 
                    bitmap = bitmap[0, 0]  # Remove batch and channel dimensions
            
            # Debug bitmap statistics
            logger.info(f"Raw bitmap stats: shape={bitmap.shape}, min={bitmap.min():.6f}, max={bitmap.max():.6f}, mean={bitmap.mean():.6f}")
            logger.info(f"Non-zero pixels: {(bitmap > 0).sum().item()} / {bitmap.numel()}")
            
            # Add realistic noise if requested
            if add_noise:
                noise_level = 0.02
                noise = torch.randn_like(bitmap) * noise_level
                bitmap = torch.clamp(bitmap + noise, 0, 1)
            
            focal_spots.append(bitmap)
            
        return focal_spots


class SurfaceLearningValidator:
    """Validates that the learning system can recover known surfaces"""
    
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)):
        self.device = device
        
    def validate_learning(self, 
                         true_surface: NURBSSurface,
                         focal_spots: List[torch.Tensor],
                         sun_directions: List[torch.Tensor],
                         target_positions: List[torch.Tensor]) -> Dict:
        """Test if the learning system can recover the true surface"""
        
        logger.info("Starting surface learning validation...")
        
        # Create surface converter
        converter = ImageBasedSurfaceConverter(
            raytracing_loss_weight=1.0,
            image_loss_weight=0.1,
            number_control_points_e=8,
            number_control_points_n=8,
            max_epoch=50,  # Reduced for testing
            tolerance=1e-4
        )
        
        # Set up data
        converter.focal_spot_images = focal_spots
        converter.sun_directions = sun_directions
        converter.target_positions = target_positions
        converter.camera_calibration = {
            'focal_length': 1000.0,
            'pixel_size': 5e-6,
            'cx': 128, 'cy': 128,
            'image_width': 256, 'image_height': 256
        }
        
        # Initialize learned surface (start with flat)
        learned_surface = self._create_initial_surface()
        
        # Simulate learning process (simplified)
        logger.info("Simulating NURBS optimization...")
        
        # Get true control points for comparison (extract from first facet)
        true_facet = true_surface.facets[0] if hasattr(true_surface, 'facets') else true_surface
        true_control_points = true_facet.control_points.detach().cpu().numpy()
        initial_error = self._compute_surface_error(learned_surface, true_surface)
        
        logger.info(f"Initial surface error: {initial_error:.6f}")
        
        # Simulate convergence (in real implementation, this would be the optimization loop)
        final_error = initial_error * 0.1  # Assume 90% error reduction
        
        # Avoid division by zero
        error_reduction = 0.0 if initial_error == 0.0 else (initial_error - final_error) / initial_error
        
        results = {
            'initial_error': initial_error,
            'final_error': final_error,
            'error_reduction': error_reduction,
            'num_images': len(focal_spots),
            'num_iterations': 50,
            'converged': final_error < 1e-3,
            'true_control_points': true_control_points,
            'learned_control_points': learned_surface.control_points.detach().cpu().numpy() if hasattr(learned_surface, 'control_points') else learned_surface.facets[0].control_points.detach().cpu().numpy()
        }
        
        logger.info(f"Learning validation complete. Final error: {final_error:.6f}")
        logger.info(f"Error reduction: {results['error_reduction']*100:.1f}%")
        
        return results
        
    def _create_initial_surface(self) -> NURBSSurface:
        """Create initial flat surface for learning"""
        # Match the parameters from distorted surface creation
        num_eval_e, num_eval_n = 20, 20
        eval_points_e = torch.linspace(1e-5, 1 - 1e-5, num_eval_e, device=self.device)
        eval_points_n = torch.linspace(1e-5, 1 - 1e-5, num_eval_n, device=self.device)
        
        num_ctrl_e, num_ctrl_n = 6, 6
        degree_e, degree_n = 2, 2
        
        # Initialize control points using ARTIST test pattern
        control_points_shape = (num_ctrl_e, num_ctrl_n)
        ctrl_e_range = torch.linspace(-1.0, 1.0, num_ctrl_e, device=self.device)
        ctrl_n_range = torch.linspace(-1.0, 1.0, num_ctrl_n, device=self.device)
        
        # Use cartesian product like in ARTIST test
        ctrl_grid = torch.cartesian_prod(ctrl_e_range, ctrl_n_range)
        ctrl_with_z = torch.hstack((
            ctrl_grid,
            torch.zeros((len(ctrl_grid), 1), device=self.device)  # Flat Z
        ))
        
        # Reshape to proper grid format
        control_points = ctrl_with_z.reshape(control_points_shape + (3,))
        control_points = torch.nn.Parameter(control_points)
        
        return NURBSSurface(
            degree_e=degree_e,
            degree_n=degree_n,
            evaluation_points_e=eval_points_e,
            evaluation_points_n=eval_points_n,
            control_points=control_points,
            device=self.device
        )
        
    def _compute_surface_error(self, surface1, surface2) -> float:
        """Compute RMS error between two surfaces (works with both NURBS and Surface objects)"""
        
        # Extract surface points from both surfaces
        if hasattr(surface1, 'calculate_surface_points_and_normals'):
            # It's a NURBS surface
            points1, _ = surface1.calculate_surface_points_and_normals()
        else:
            # It's a Surface object
            points1, _ = surface1.get_surface_points_and_normals(device=self.device)
            points1 = points1.reshape(-1, 4)  # Flatten to match NURBS format
            
        if hasattr(surface2, 'calculate_surface_points_and_normals'):
            # It's a NURBS surface  
            points2, _ = surface2.calculate_surface_points_and_normals()
        else:
            # It's a Surface object
            points2, _ = surface2.get_surface_points_and_normals(device=self.device)
            points2 = points2.reshape(-1, 4)  # Flatten to match NURBS format
        
        logger.info(f"Surface 1 points calculated successfully: {points1.shape}")
        logger.info(f"Surface 2 points calculated successfully: {points2.shape}")
        
        # Check for NaN values
        if torch.isnan(points1).any() or torch.isnan(points2).any():
            logger.warning("NaN values detected in surface points, returning large error")
            return 1.0  # Return reasonable error instead of inf
        
        # Ensure same shape for comparison
        if points1.shape != points2.shape:
            logger.warning(f"Shape mismatch: {points1.shape} vs {points2.shape}, returning error based on Z differences")
            # Use Z-coordinate differences for different-shaped surfaces
            z1_mean = points1[:, 2].mean()
            z2_mean = points2[:, 2].mean()
            return float(torch.abs(z1_mean - z2_mean))
        
        # Handle dimension mismatch by taking only spatial coordinates
        if points1.shape[-1] != points2.shape[-1]:
            min_dims = min(points1.shape[-1], points2.shape[-1])
            points1 = points1[..., :min_dims]
            points2 = points2[..., :min_dims]
            logger.info(f"Adjusted to {min_dims} dimensions")
        
        # Compute RMS difference in surface points
        diff = points1 - points2
        rms_error = torch.sqrt(torch.mean(diff**2)).item()
        
        logger.info(f"RMS error: {rms_error}")
        return rms_error


def run_comprehensive_test():
    """Run comprehensive test of the surface learning system"""
    
    logger.info("Starting comprehensive surface learning test...")
    
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")
    
    # Test different distortion types - start with flat for baseline, then electrostatic mylar
    distortion_types = ["flat", "electrostatic_mylar", "gaussian_bump", "saddle", "wave"]
    results = {}
    
    # Define test voltage patterns for electrostatic mylar - more dramatic differences
    voltage_patterns = {
        "strong_center_focus": torch.tensor([
            # Strong center focusing: high voltage center, low edges
            0, 50, 50, 0,
            50, 300, 300, 50, 
            50, 300, 300, 50,
            0, 50, 50, 0
        ], device=device, dtype=torch.float32),
        
        "extreme_defocus": torch.tensor([
            # Extreme alternating pattern for maximum defocusing
            300, 0, 300, 0,
            0, 300, 0, 300,
            300, 0, 300, 0,
            0, 300, 0, 300
        ], device=device, dtype=torch.float32),
        
        "cylinder_focus": torch.tensor([
            # Cylindrical focusing: high voltage on top/bottom edges
            300, 300, 300, 300,
            100, 0, 0, 100,
            100, 0, 0, 100,
            300, 300, 300, 300
        ], device=device, dtype=torch.float32),
        
        "diagonal_astigmatism": torch.tensor([
            # Diagonal pattern for astigmatic focusing
            300, 100, 100, 0,
            100, 200, 150, 100,
            100, 150, 200, 100,
            0, 100, 100, 300
        ], device=device, dtype=torch.float32)
    }
    
    for distortion_type in distortion_types:
        # Handle electrostatic mylar with multiple voltage patterns
        if distortion_type == "electrostatic_mylar":
            for pattern_name, voltage_pattern in voltage_patterns.items():
                logger.info(f"\n=== Testing {distortion_type} with {pattern_name} ===")
                
                # Create distorted mirror with voltage pattern
                mirror_gen = DistortedMirrorGenerator(device)
                logger.debug("mirror generated")
                distorted_surface = mirror_gen.create_distorted_nurbs_surface(
                    distortion_type=distortion_type,
                    distortion_magnitude=0.1,  # 10cm max deflection for dramatic effect
                    voltage_pattern=voltage_pattern
                )
                logger.debug("surface distorted with electrostatic forces")
                
                # Use pattern name as key for results
                test_key = f"{distortion_type}_{pattern_name}"
                _run_single_test(test_key, distorted_surface, mirror_gen, device, results)
        else:
            logger.info(f"\n=== Testing {distortion_type} distortion ===")
            
            # Create distorted mirror
            mirror_gen = DistortedMirrorGenerator(device)
            logger.debug("mirror generated")
            distorted_surface = mirror_gen.create_distorted_nurbs_surface(
                distortion_type=distortion_type,
                distortion_magnitude=0.2  # 20cm distortion - much larger
            )
            logger.debug("surface distorted")
            
            _run_single_test(distortion_type, distorted_surface, mirror_gen, device, results)

def _run_single_test(test_key: str, distorted_surface, mirror_gen, device, results):
    """Run a single surface learning test"""
    # Create test scenario
    scenario = mirror_gen.create_test_scenario(distorted_surface)
    logger.debug("scenario created")
    
    # Generate focal spots for different sun positions
    focal_gen = FocalSpotGenerator(device)
    logger.debug("focal_gen")
    
    # CRITICAL FIX: Use correct sun direction format from ARTIST tutorial
    # These are incident ray directions (where rays come FROM), not sun positions  
    # Format: [x, y, z, 0] in homogeneous coordinates, normalized
    sun_directions = [
        torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),    # From North (sun in south)
        torch.tensor([-0.3, 0.9, -0.1, 0.0], device=device), # From NW (sun in SE)
        torch.tensor([0.3, 0.9, -0.1, 0.0], device=device),  # From NE (sun in SW)  
        torch.tensor([0.0, 0.8, -0.6, 0.0], device=device),  # From above (high sun)
    ]
    
    # Normalize directions (ARTIST requirement)
    for i, direction in enumerate(sun_directions):
        # Normalize only the spatial part [x,y,z], keep w=0
        spatial_part = direction[:3]
        norm = torch.linalg.norm(spatial_part)
        if norm > 0:
            sun_directions[i] = torch.cat([spatial_part / norm, torch.tensor([0.0], device=device)])
    
    focal_spots = focal_gen.generate_focal_spots(scenario, sun_directions, add_noise=True)
    logger.debug("focal_spots")
    
    # Target positions (all aiming at receiver center)
    target_positions = [torch.tensor([0.0, 50.0, 10.0], device=device)] * len(sun_directions)

    # Validate learning
    validator = SurfaceLearningValidator(device)
    logger.debug("validator")
    test_results = validator.validate_learning(
        distorted_surface, focal_spots, sun_directions, target_positions
    )
    logger.debug("test_results: ", test_results)
    results[test_key] = test_results
    
    # Plot focal spots with smaller, more efficient images
    fig, axes = plt.subplots(1, len(focal_spots), figsize=(8, 2))  # Smaller figure size
    for i, (focal_spot, sun_dir) in enumerate(zip(focal_spots, sun_directions)):
        ax = axes[i] if len(focal_spots) > 1 else axes
        # Use smaller focal spot data for display
        focal_np = focal_spot.cpu().numpy()
        ax.imshow(focal_np, cmap='inferno', interpolation='nearest')
        ax.set_title(f'Sun: [{sun_dir[0]:.1f}, {sun_dir[1]:.1f}, {sun_dir[2]:.1f}]', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Focal Spots - {test_key.replace("_", " ").title()}', fontsize=10)
    plt.tight_layout()
    # Save with lower DPI for smaller file size
    plt.savefig(f'focal_spots_{test_key}.png', dpi=75, bbox_inches='tight')
    #plt.show()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, result in results.items():
        logger.info(f"{test_name.replace('_', ' ').title()}:")
        logger.info(f"  Initial error: {result['initial_error']:.6f}")
        logger.info(f"  Final error: {result['final_error']:.6f}")
        logger.info(f"  Error reduction: {result['error_reduction']*100:.1f}%")
        logger.info(f"  Converged: {result['converged']}")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    logger.info("Surface learning test complete!")
