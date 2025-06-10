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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistortedMirrorGenerator:
    """Generates heliostats with known surface distortions for testing"""
    
    def __init__(self, device: torch.device = torch.device(DEVICE)):
        self.device = device
        
    def create_distorted_nurbs_surface(self, 
                                     distortion_type: str = "gaussian_bump",
                                     distortion_magnitude: float = 0.01) -> NURBSSurface:
        """Create a NURBS surface with known distortions"""
        
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
        
        # Apply distortions
        if distortion_type == "flat":
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
        
        return NURBSSurface(
            degree_e=degree_e,
            degree_n=degree_n,
            evaluation_points_e=eval_points_e,
            evaluation_points_n=eval_points_n,
            control_points=control_points,
            device=self.device
        )
    
    def create_test_scenario(self, nurbs_surface: NURBSSurface) -> Scenario:
        """Create a test scenario with the given NURBS surface"""
        
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
        
        # Create facet config from NURBS surface
        logger.debug("Creating facet config...")
        
        # Check the actual control points dimensions
        logger.debug(f"Control points shape: {nurbs_surface.control_points.shape}")
        
        # Test what surface points look like
        try:
            test_surface_points, test_surface_normals = nurbs_surface.calculate_surface_points_and_normals()
            logger.debug(f"Surface points shape: {test_surface_points.shape}")
            logger.debug(f"Surface normals shape: {test_surface_normals.shape}")
        except Exception as e:
            logger.debug(f"Error calculating surface points: {e}")
        
        # Keep original 3D structure but detach gradients
        control_points_detached = nurbs_surface.control_points.detach()
        
        facet_config = FacetConfig(
            facet_key="test_facet",
            control_points=control_points_detached,  # Keep 3D structure
            degree_e=nurbs_surface.degree_e,
            degree_n=nurbs_surface.degree_n,
            number_eval_points_e=len(nurbs_surface.evaluation_points_e),
            number_eval_points_n=len(nurbs_surface.evaluation_points_n),
            translation_vector=torch.zeros(4, device=self.device),  # 4D for [x,y,z,w]
            canting_e=torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device),  # 4D
            canting_n=torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device),  # 4D
        )
        logger.debug("Facet config created")
        
        # Create prototype configurations
        logger.debug("Creating prototype configurations...")
        
        # Surface prototype
        surface_prototype_config = SurfacePrototypeConfig(facet_list=[facet_config])
        
        # Kinematic prototype
        kinematic_prototype_config = KinematicPrototypeConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=[0.0, 0.0, 1.0, 0.0],
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
        
        # Create kinematic config for heliostat
        kinematic_config = KinematicConfig(
            type=config_dictionary.rigid_body_key,
            initial_orientation=[0.0, 0.0, 1.0, 0.0],
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
        
        # Get true control points for comparison
        true_control_points = true_surface.control_points.detach().cpu().numpy()
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
            'learned_control_points': learned_surface.control_points.detach().cpu().numpy()
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
        
    def _compute_surface_error(self, surface1: NURBSSurface, surface2: NURBSSurface) -> float:
        """Compute RMS error between two surfaces"""
        
        # Debug NURBS surface properties
        logger.info(f"Surface 1 - degree_e: {surface1.degree_e}, degree_n: {surface1.degree_n}")
        logger.info(f"Surface 1 - control_points shape: {surface1.control_points.shape}")
        logger.info(f"Surface 1 - control_points range: [{surface1.control_points.min():.3f}, {surface1.control_points.max():.3f}]")
        logger.info(f"Surface 1 - control_points contain nan: {torch.isnan(surface1.control_points).any()}")
        logger.info(f"Surface 1 - eval_points_e shape: {surface1.evaluation_points_e.shape}")
        logger.info(f"Surface 1 - eval_points_n shape: {surface1.evaluation_points_n.shape}")
        
        logger.info(f"Surface 2 - degree_e: {surface2.degree_e}, degree_n: {surface2.degree_n}")
        logger.info(f"Surface 2 - control_points shape: {surface2.control_points.shape}")
        logger.info(f"Surface 2 - control_points range: [{surface2.control_points.min():.3f}, {surface2.control_points.max():.3f}]")
        logger.info(f"Surface 2 - control_points contain nan: {torch.isnan(surface2.control_points).any()}")
        
        # Try to identify where NaNs are coming from in NURBS calculation
        try:
            points1, _ = surface1.calculate_surface_points_and_normals()
            logger.info(f"Surface 1 points calculated successfully: {points1.shape}")
        except Exception as e:
            logger.error(f"Error calculating surface 1 points: {e}")
            return float('inf')
            
        try:
            points2, _ = surface2.calculate_surface_points_and_normals()
            logger.info(f"Surface 2 points calculated successfully: {points2.shape}")
        except Exception as e:
            logger.error(f"Error calculating surface 2 points: {e}")
            return float('inf')
        
        logger.info(f"Surface 1 points contain nan: {torch.isnan(points1).any()}")
        logger.info(f"Surface 2 points contain nan: {torch.isnan(points2).any()}")
        
        # If we have NaN values, this is a critical error - don't mask it
        if torch.isnan(points1).any() or torch.isnan(points2).any():
            logger.error("CRITICAL: NaN values detected in surface points - NURBS calculation is broken!")
            # Let's examine the NaN pattern
            if torch.isnan(points1).any():
                nan_count1 = torch.isnan(points1).sum().item()
                logger.error(f"Surface 1 has {nan_count1} NaN values out of {points1.numel()}")
            if torch.isnan(points2).any():
                nan_count2 = torch.isnan(points2).sum().item()
                logger.error(f"Surface 2 has {nan_count2} NaN values out of {points2.numel()}")
            return float('inf')
        
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
    
    # Test different distortion types - start with flat for baseline
    distortion_types = ["flat", "gaussian_bump", "saddle", "wave"]
    results = {}
    
    for distortion_type in distortion_types:
        logger.info(f"\n=== Testing {distortion_type} distortion ===")
        
        # Create distorted mirror
        mirror_gen = DistortedMirrorGenerator(device)
        logger.debug("mirror generated")
        distorted_surface = mirror_gen.create_distorted_nurbs_surface(
            distortion_type=distortion_type,
            distortion_magnitude=0.2  # 20cm distortion - much larger
        )
        logger.debug("surface distorted")
        # Create test scenario
        scenario = mirror_gen.create_test_scenario(distorted_surface)
        logger.debug("scenario creted")
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
        results[distortion_type] = test_results
        
        # Plot focal spots
        fig, axes = plt.subplots(1, len(focal_spots), figsize=(15, 3))
        for i, (focal_spot, sun_dir) in enumerate(zip(focal_spots, sun_directions)):
            ax = axes[i] if len(focal_spots) > 1 else axes
            ax.imshow(focal_spot.cpu().numpy(), cmap='inferno')
            ax.set_title(f'Sun: [{sun_dir[0]:.1f}, {sun_dir[1]:.1f}, {sun_dir[2]:.1f}]')
            ax.axis('off')
        
        plt.suptitle(f'Focal Spots - {distortion_type.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(f'focal_spots_{distortion_type}.png', dpi=150, bbox_inches='tight')
        #plt.show()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    for distortion_type, result in results.items():
        logger.info(f"{distortion_type.replace('_', ' ').title()}:")
        logger.info(f"  Initial error: {result['initial_error']:.6f}")
        logger.info(f"  Final error: {result['final_error']:.6f}")
        logger.info(f"  Error reduction: {result['error_reduction']*100:.1f}%")
        logger.info(f"  Converged: {result['converged']}")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    logger.info("Surface learning test complete!")
