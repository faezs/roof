#!/usr/bin/env python3
"""
Test Kinematic Alignment Script

This script tests the proper kinematic alignment workflow extracted from the ARTIST tutorials.
It's a minimal test to verify that heliostat alignment is working correctly.
"""

import logging
import pathlib
import torch
import matplotlib.pyplot as plt

# Force MPS device usage
DEVICE = "mps"

# ARTIST imports
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util.scenario_generator import ScenarioGenerator
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_test_scenario():
    """Create a minimal test scenario for kinematic alignment testing"""
    
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")
    
    # Create simple flat NURBS surface
    num_eval_e, num_eval_n = 10, 10
    eval_points_e = torch.linspace(1e-5, 1 - 1e-5, num_eval_e, device=device)
    eval_points_n = torch.linspace(1e-5, 1 - 1e-5, num_eval_n, device=device)
    
    # Simple 3x3 control points grid, degree 2
    num_ctrl_e, num_ctrl_n = 3, 3
    degree_e, degree_n = 2, 2
    
    # Create flat control points
    ctrl_e_range = torch.linspace(-1.0, 1.0, num_ctrl_e, device=device)
    ctrl_n_range = torch.linspace(-1.0, 1.0, num_ctrl_n, device=device)
    ctrl_grid = torch.cartesian_prod(ctrl_e_range, ctrl_n_range)
    ctrl_with_z = torch.hstack((
        ctrl_grid,
        torch.zeros((len(ctrl_grid), 1), device=device)  # Flat surface
    ))
    control_points = ctrl_with_z.reshape((num_ctrl_e, num_ctrl_n, 3))
    
    # Create facet config
    facet_config = FacetConfig(
        facet_key="test_facet",
        control_points=control_points,
        degree_e=degree_e,
        degree_n=degree_n,
        number_eval_points_e=num_eval_e,
        number_eval_points_n=num_eval_n,
        translation_vector=torch.zeros(4, device=device),
        canting_e=torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device),
        canting_n=torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device),
    )
    
    # Create target area (receiver)
    target_config = TargetAreaConfig(
        target_area_key="receiver",
        geometry="planar",
        center=torch.tensor([0.0, 50.0, 10.0, 1.0], device=device),
        normal_vector=torch.tensor([0.0, -1.0, 0.0, 0.0], device=device),
        plane_e=5.0,  # 5m x 5m receiver
        plane_u=5.0,
    )
    target_list_config = TargetAreaListConfig(target_area_list=[target_config])
    
    # Create light source (sun)
    light_source_config = LightSourceConfig(
        light_source_key="sun",
        light_source_type="sun",
        number_of_rays=5000,
        distribution_type="normal",
        mean=0.0,
        covariance=2.0e-05,
    )
    light_source_list_config = LightSourceListConfig(light_source_list=[light_source_config])
    
    # Create prototype configurations
    surface_prototype_config = SurfacePrototypeConfig(facet_list=[facet_config])
    kinematic_prototype_config = KinematicPrototypeConfig(
        type=config_dictionary.rigid_body_key,
        initial_orientation=[0.0, 0.0, 1.0, 0.0],
    )
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
    prototype_config = PrototypeConfig(
        surface_prototype=surface_prototype_config,
        kinematic_prototype=kinematic_prototype_config,
        actuators_prototype=actuator_prototype_config,
    )
    
    # Create heliostat config
    surface_config = SurfaceConfig(facet_list=[facet_config])
    kinematic_config = KinematicConfig(
        type=config_dictionary.rigid_body_key,
        initial_orientation=[0.0, 0.0, 1.0, 0.0],
    )
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
    
    heliostat_config = HeliostatConfig(
        name="test_heliostat",
        id=1,
        position=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        aim_point=torch.tensor([0.0, 50.0, 10.0, 1.0], device=device),
        surface=surface_config,
        kinematic=kinematic_config,
        actuators=actuator_list_config,
    )
    heliostat_list_config = HeliostatListConfig(heliostat_list=[heliostat_config])
    
    # Create power plant config
    power_plant_config = PowerPlantConfig(
        power_plant_position=torch.tensor([0.0, 0.0, 0.0], device=device),
    )
    
    # Generate scenario
    temp_dir = pathlib.Path("/tmp")
    scenario_file = temp_dir / "alignment_test_scenario"
    
    scenario_generator = ScenarioGenerator(
        file_path=scenario_file,
        power_plant_config=power_plant_config,
        target_area_list_config=target_list_config,
        light_source_list_config=light_source_list_config,
        heliostat_list_config=heliostat_list_config,
        prototype_config=prototype_config,
    )
    
    scenario_generator.generate_scenario()
    
    # Load scenario
    import h5py
    scenario_h5_path = temp_dir / (scenario_file.name + ".h5")
    
    with h5py.File(scenario_h5_path) as scenario_file_h5:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file_h5,
            device=device
        )
    
    return scenario


def test_kinematic_alignment():
    """Test the kinematic alignment workflow from ARTIST tutorials"""
    
    logger.info("Creating test scenario...")
    scenario = create_simple_test_scenario()
    device = torch.device(DEVICE)
    
    logger.info("Setting up raytracer...")
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=scenario.heliostat_field.heliostat_groups[0],
    )
    
    # Test different sun directions - incident ray directions (from sun to earth)
    sun_directions = [
        torch.tensor([0.0, -1.0, 0.0, 0.0], device=device),   # Sun in south (rays from south)
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),    # Sun in east (rays from east) 
        torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device),   # Sun in west (rays from west)
        torch.tensor([0.0, 0.0, -1.0, 0.0], device=device),   # Sun above (rays from above)
    ]
    
    direction_names = ["South", "East", "West", "Above"]
    focal_spots = []
    
    # Set up fixed parameters
    active_heliostats_mask = torch.tensor([1], dtype=torch.int32, device=device)
    target_area_mask = torch.tensor([0], device=device)
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    
    for i, (sun_dir, name) in enumerate(zip(sun_directions, direction_names)):
        logger.info(f"Testing alignment for sun direction: {name} {sun_dir.cpu().numpy()}")
        
        # STEP 1: Activate heliostats (CRITICAL!)
        heliostat_group.activate_heliostats(active_heliostats_mask=active_heliostats_mask)
        
        # STEP 2: Align heliostat surfaces (KEY KINEMATIC STEP!)
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=sun_dir.unsqueeze(0),  # Must be batched
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )
        
        # STEP 3: Perform raytracing
        logger.info(f"  About to trace rays with sun_dir: {sun_dir}")
        logger.info(f"  Target center: {scenario.target_areas.centers[target_area_mask]}")
        logger.info(f"  Heliostat position: {scenario.heliostat_field.heliostat_groups[0].positions}")
        
        bitmap = raytracer.trace_rays(
            incident_ray_directions=sun_dir.unsqueeze(0),  # Must be batched
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )
        
        logger.info(f"  Raytracing completed")
        
        # Handle output format
        if isinstance(bitmap, torch.Tensor):
            if bitmap.dim() == 3:  # [batch, height, width]
                bitmap = bitmap[0]
            elif bitmap.dim() == 4:  # [batch, channels, height, width]
                bitmap = bitmap[0, 0]
        
        # Log statistics
        logger.info(f"  Bitmap shape: {bitmap.shape}")
        logger.info(f"  Min/Max: {bitmap.min():.6f} / {bitmap.max():.6f}")
        logger.info(f"  Mean: {bitmap.mean():.6f}")
        logger.info(f"  Non-zero pixels: {(bitmap > 0).sum().item()} / {bitmap.numel()}")
        
        focal_spots.append(bitmap.cpu().numpy())
    
    # Plot results
    fig, axes = plt.subplots(1, len(focal_spots), figsize=(15, 3))
    if len(focal_spots) == 1:
        axes = [axes]
    
    for i, (focal_spot, name) in enumerate(zip(focal_spots, direction_names)):
        axes[i].imshow(focal_spot, cmap='inferno')
        axes[i].set_title(f'Focal Spot - Sun {name}')
        axes[i].axis('off')
    
    plt.suptitle('Kinematic Alignment Test - Focal Spots')
    plt.tight_layout()
    plt.savefig('/Users/faezs/roof/kinematic_alignment_test.png', dpi=150, bbox_inches='tight')
    logger.info("Saved test results to kinematic_alignment_test.png")
    
    # Check if alignment is working
    success_count = sum(1 for spot in focal_spots if spot.max() > 0.01)
    logger.info(f"Successful focal spots: {success_count} / {len(focal_spots)}")
    
    if success_count > 0:
        logger.info("✅ Kinematic alignment is working! Focal spots generated successfully.")
        return True
    else:
        logger.error("❌ Kinematic alignment failed! No focal spots generated.")
        return False


if __name__ == "__main__":
    success = test_kinematic_alignment()
    if success:
        print("Kinematic alignment test PASSED!")
    else:
        print("Kinematic alignment test FAILED!")