"""
Image-Based Surface Converter for ARTIST Integration

This module implements surface reconstruction from heliostat focal spot images,
following ARTIST design patterns and integrating with the verified control system.
"""

import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import device utilities
from device_utils import get_default_device, ensure_device

# Get default device for the session
DEFAULT_DEVICE = get_default_device()

# ARTIST imports (adjust paths as needed)
from artist.util.surface_converter import SurfaceConverter
from artist.util.configuration_classes import FacetConfig
from artist.util.nurbs import NURBSSurface
from artist.util import utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario

log = logging.getLogger(__name__)
"""A logger for the image-based surface converter."""


class ImageBasedSurfaceConverter(SurfaceConverter):
    """
    Surface converter that learns NURBS surfaces from heliostat focal spot images.
    
    This converter follows ARTIST patterns while adding computer vision capabilities
    to reconstruct heliostat surfaces from real-time focal spot observations instead
    of requiring deflectometry data.
    
    Key Innovation: Uses raytracing forward model to match predicted vs observed 
    focal spots, optimizing NURBS control points through gradient descent.
    
    Attributes
    ----------
    focal_spot_images : List[torch.Tensor]
        Observed focal spot images from camera
    sun_directions : List[torch.Tensor] 
        Incident sun direction for each image (3D unit vectors)
    target_positions : List[torch.Tensor]
        Target center position for each focal spot image
    camera_calibration : Dict
        Camera intrinsic and extrinsic parameters
    raytracing_loss_weight : float
        Weight for raytracing-based loss vs image similarity
    image_loss_weight : float  
        Weight for direct image comparison loss
    convergence_threshold : float
        Convergence criteria for optimization
    """
    
    def __init__(
        self,
        focal_spot_images: Optional[List[torch.Tensor]] = None,
        sun_directions: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[List[torch.Tensor]] = None,
        camera_calibration: Optional[Dict] = None,
        raytracing_loss_weight: float = 1.0,
        image_loss_weight: float = 0.1,
        convergence_threshold: float = 1e-6,
        # Inherit all SurfaceConverter parameters
        **kwargs
    ) -> None:
        """
        Initialize the image-based surface converter.
        
        Parameters
        ----------
        focal_spot_images : List[torch.Tensor], optional
            List of observed focal spot images (each as 2D tensor)
        sun_directions : List[torch.Tensor], optional  
            Incident sun directions for each image (3D unit vectors)
        target_positions : List[torch.Tensor], optional
            Target center positions for focal spots
        camera_calibration : Dict, optional
            Camera calibration parameters
        raytracing_loss_weight : float
            Weight for raytracing loss component (default: 1.0)
        image_loss_weight : float
            Weight for image similarity loss (default: 0.1)
        convergence_threshold : float
            Optimization convergence threshold (default: 1e-6)
        **kwargs
            Additional parameters passed to SurfaceConverter
        """
        super().__init__(**kwargs)
        
        self.focal_spot_images = focal_spot_images or []
        self.sun_directions = sun_directions or []
        self.target_positions = target_positions or []
        self.camera_calibration = camera_calibration or {}
        
        self.raytracing_loss_weight = raytracing_loss_weight
        self.image_loss_weight = image_loss_weight
        self.convergence_threshold = convergence_threshold
        
        # Image preprocessing parameters
        self.image_noise_threshold = 0.01
        self.flux_calibration_factor = 1000.0  # W/m² per image intensity unit
        
    def add_focal_spot_observation(
        self,
        image: torch.Tensor,
        sun_direction: torch.Tensor,
        target_position: torch.Tensor
    ) -> None:
        """
        Add a new focal spot observation to the dataset.
        
        Parameters
        ----------
        image : torch.Tensor
            Focal spot image (2D tensor)
        sun_direction : torch.Tensor  
            Sun direction (3D unit vector)
        target_position : torch.Tensor
            Target center position (3D coordinates)
        """
        self.focal_spot_images.append(image.clone())
        self.sun_directions.append(sun_direction.clone())
        self.target_positions.append(target_position.clone())
        
    def preprocess_focal_spot_image(
        self, 
        image: torch.Tensor,
        device: Union[torch.device, str] = None
    ) -> torch.Tensor:
        """
        Preprocess focal spot image for analysis.
        
        Parameters
        ----------
        image : torch.Tensor
            Raw focal spot image
        device : Union[torch.device, str]
            Device for tensor operations
            
        Returns
        -------
        torch.Tensor
            Preprocessed image with noise reduction and calibration
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        image = image.to(device)
        
        # Noise reduction
        image = torch.where(image < self.image_noise_threshold, 
                           torch.zeros_like(image), image)
        
        # Normalize to [0,1] range
        if image.max() > 0:
            image = image / image.max()
            
        # Apply flux calibration
        image = image * self.flux_calibration_factor
        
        # Gaussian smoothing to reduce noise
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            image = F.gaussian_blur(image, kernel_size=3, sigma=1.0)
            image = image.squeeze(0).squeeze(0)  # Remove extra dims
            
        return image
        
    def extract_focal_spot_center(
        self, 
        image: torch.Tensor,
        device: Union[torch.device, str] = None
    ) -> torch.Tensor:
        """
        Extract center of mass from focal spot image.
        
        This follows ARTIST's pattern used in AlignmentOptimizer.
        
        Parameters
        ----------
        image : torch.Tensor
            Focal spot image
        device : Union[torch.device, str]
            Device for computation
            
        Returns
        -------
        torch.Tensor
            Center of mass coordinates (2D)
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        image = image.to(device)
        
        # Create coordinate grids
        h, w = image.shape
        y_coords = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1)
        x_coords = torch.arange(w, dtype=torch.float32, device=device).view(1, -1)
        
        # Calculate weighted center of mass
        total_intensity = image.sum()
        if total_intensity > 0:
            center_x = (image * x_coords).sum() / total_intensity
            center_y = (image * y_coords).sum() / total_intensity
        else:
            center_x = torch.tensor(w / 2.0, device=device)
            center_y = torch.tensor(h / 2.0, device=device)
            
        return torch.stack([center_x, center_y])
        
    def image_to_world_coordinates(
        self,
        image_coords: torch.Tensor,
        target_position: torch.Tensor,
        device: Union[torch.device, str] = None
    ) -> torch.Tensor:
        """
        Convert image coordinates to 3D world coordinates on target plane.
        
        Parameters
        ----------
        image_coords : torch.Tensor
            2D image coordinates
        target_position : torch.Tensor
            3D target center position
        device : Union[torch.device, str]
            Device for computation
            
        Returns
        -------
        torch.Tensor
            3D world coordinates
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        
        # Simple projection model - can be enhanced with full camera calibration
        # Assumes target plane is perpendicular to camera view
        
        if 'focal_length' in self.camera_calibration:
            fx = self.camera_calibration['focal_length']
            fy = self.camera_calibration['focal_length'] 
        else:
            fx = fy = 1000.0  # Default focal length
            
        if 'pixel_size' in self.camera_calibration:
            px = self.camera_calibration['pixel_size']
            py = self.camera_calibration['pixel_size']
        else:
            px = py = 1e-5  # Default pixel size in meters
            
        # Convert to metric coordinates
        x_metric = (image_coords[0] * px - self.camera_calibration.get('cx', 0)) * target_position[2] / fx
        y_metric = (image_coords[1] * py - self.camera_calibration.get('cy', 0)) * target_position[2] / fy
        
        world_coords = torch.tensor([
            target_position[0] + x_metric,
            target_position[1] + y_metric, 
            target_position[2]
        ], device=device)
        
        return world_coords
        
    def fit_nurbs_from_images(
        self,
        scenario: Scenario,
        heliostat_id: int = 0,
        device: Union[torch.device, str] = None
    ) -> NURBSSurface:
        """
        Fit NURBS surface by matching predicted vs observed focal spots.
        
        This is the core innovation - uses ARTIST's raytracing forward model
        to optimize NURBS parameters against real focal spot observations.
        
        Parameters
        ----------
        scenario : Scenario
            ARTIST scenario containing heliostat and targets
        heliostat_id : int
            Index of heliostat to optimize (default: 0)
        device : Union[torch.device, str]
            Device for computation
            
        Returns
        -------
        NURBSSurface
            Optimized NURBS surface
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        log.info("Starting NURBS fitting from focal spot images")
        
        if len(self.focal_spot_images) == 0:
            raise ValueError("No focal spot images provided")
            
        # Initialize NURBS surface with reasonable control points
        nurbs_surface = self._initialize_nurbs_surface(device)
        
        # Setup optimizer following ARTIST patterns
        optimizer = Adam([nurbs_surface.control_points], lr=self.initial_learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=50,
            threshold=1e-7, threshold_mode='abs'
        )
        
        # Get heliostat from scenario
        heliostat = scenario.heliostats.heliostat_list[heliostat_id]
        
        loss = torch.inf
        epoch = 0
        
        log.info(f"Optimizing NURBS with {len(self.focal_spot_images)} focal spot observations")
        
        while loss > self.convergence_threshold and epoch <= self.max_epoch:
            total_loss = torch.tensor(0.0, device=device)
            
            for i, (image, sun_dir, target_pos) in enumerate(zip(
                self.focal_spot_images, self.sun_directions, self.target_positions
            )):
                # Move tensors to device
                image = image.to(device)
                sun_dir = sun_dir.to(device) 
                target_pos = target_pos.to(device)
                
                # Update heliostat surface with current NURBS
                self._update_heliostat_surface(heliostat, nurbs_surface, device)
                
                # Align heliostat for current sun direction
                heliostat.set_aligned_surface_with_incident_ray_direction(
                    incident_ray_direction=sun_dir, device=device
                )
                
                # Run raytracing to predict focal spot
                predicted_center = self._raytrace_focal_spot(
                    scenario, heliostat_id, sun_dir, target_pos, device
                )
                
                # Extract observed focal spot center
                processed_image = self.preprocess_focal_spot_image(image, device)
                observed_center_2d = self.extract_focal_spot_center(processed_image, device)
                observed_center_3d = self.image_to_world_coordinates(
                    observed_center_2d, target_pos, device
                )
                
                # Calculate loss (center of mass difference)
                raytracing_loss = (predicted_center - observed_center_3d).norm()
                
                # Optional: Add image similarity loss
                image_loss = torch.tensor(0.0, device=device)
                if self.image_loss_weight > 0:
                    # This would require generating predicted image from raytracing bitmap
                    # For now, focus on center-of-mass matching
                    pass
                    
                loss_i = (self.raytracing_loss_weight * raytracing_loss + 
                         self.image_loss_weight * image_loss)
                total_loss += loss_i
                
            # Average loss over all observations
            loss = total_loss / len(self.focal_spot_images)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 100 == 0:
                log.info(f"Epoch: {epoch}, Loss: {loss.item():.6f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                        
            epoch += 1
            
        log.info(f"NURBS optimization completed after {epoch} epochs, "
                f"final loss: {loss.item():.6f}")
        
        return nurbs_surface
        
    def _initialize_nurbs_surface(self, device: Union[torch.device, str]) -> NURBSSurface:
        """Initialize NURBS surface with reasonable default control points."""
        device = ensure_device(device, DEFAULT_DEVICE)
        
        # Create control points for flat surface initially
        control_points_shape = (self.number_control_points_e, self.number_control_points_n)
        control_points = torch.zeros(control_points_shape + (3,), device=device)
        
        # Set up grid of control points
        width = 2.0  # meters - typical heliostat width
        height = 1.0  # meters - typical heliostat height
        
        x_coords = torch.linspace(-width/2, width/2, self.number_control_points_e, device=device)
        y_coords = torch.linspace(-height/2, height/2, self.number_control_points_n, device=device)
        
        for i in range(self.number_control_points_e):
            for j in range(self.number_control_points_n):
                control_points[i, j, 0] = x_coords[i]
                control_points[i, j, 1] = y_coords[j]
                control_points[i, j, 2] = 0.0  # Start flat
                
        # Make control points learnable
        control_points = torch.nn.Parameter(control_points)
        
        # Create evaluation points
        eval_points_e = torch.linspace(0, 1, 100, device=device)
        eval_points_n = torch.linspace(0, 1, 100, device=device)
        
        # Create NURBS surface
        nurbs_surface = NURBSSurface(
            degree_e=self.degree_e,
            degree_n=self.degree_n,
            evaluation_points_e=eval_points_e,
            evaluation_points_n=eval_points_n,
            control_points=control_points,
            device=device
        )
        
        return nurbs_surface
        
    def _update_heliostat_surface(
        self, 
        heliostat, 
        nurbs_surface: NURBSSurface, 
        device: Union[torch.device, str]
    ) -> None:
        """Update heliostat surface with new NURBS parameters."""
        # Generate surface points and normals from NURBS
        surface_points, surface_normals = nurbs_surface.calculate_surface_points_and_normals(device)
        
        # Update heliostat surface data
        heliostat.surface_points = surface_points
        heliostat.surface_normals = surface_normals
        heliostat.is_aligned = False  # Force re-alignment
        
    def _raytrace_focal_spot(
        self,
        scenario: Scenario,
        heliostat_id: int,
        sun_direction: torch.Tensor,
        target_position: torch.Tensor,
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        """
        Raytrace heliostat to predict focal spot center.
        
        Follows ARTIST's AlignmentOptimizer pattern for raytracing.
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        
        # Find target area containing the target position
        target_area = None
        for area in scenario.target_areas.target_area_list:
            # Simple distance check - can be improved
            dist = torch.norm(area.center - target_position)
            if dist < 5.0:  # Within 5m of target center
                target_area = area
                break
                
        if target_area is None:
            # Create temporary target area
            target_area_name = "temp_target"
        else:
            target_area_name = target_area.name
            
        # Create raytracer
        raytracer = HeliostatRayTracer(
            scenario=scenario,
            aim_point_area=target_area_name,
            batch_size=self.batch_size if hasattr(self, 'batch_size') else 1000,
            random_seed=42
        )
        
        # Trace rays
        flux_bitmap = raytracer.trace_rays(
            incident_ray_direction=sun_direction,
            device=device
        )
        
        # Normalize bitmap
        flux_bitmap = raytracer.normalize_bitmap(flux_bitmap)
        
        # Extract center of mass using ARTIST's utility
        predicted_center = utils.get_center_of_mass(
            bitmap=flux_bitmap,
            target_center=target_position,
            plane_e=target_area.plane_e if target_area else torch.tensor([1,0,0], device=device),
            plane_u=target_area.plane_u if target_area else torch.tensor([0,1,0], device=device),
            device=device
        )
        
        return predicted_center
        
    def generate_surface_config_from_images(
        self,
        scenario: Scenario,
        heliostat_id: int = 0,
        device: Union[torch.device, str] = None
    ) -> List[FacetConfig]:
        """
        Generate surface configuration from focal spot images.
        
        This is the main entry point that follows ARTIST's SurfaceConverter pattern.
        
        Parameters
        ----------
        scenario : Scenario
            ARTIST scenario containing heliostat setup
        heliostat_id : int
            Heliostat to optimize (default: 0)
        device : Union[torch.device, str]
            Device for computation
            
        Returns
        -------
        List[FacetConfig]
            Facet configurations compatible with ARTIST
        """
        device = ensure_device(device, DEFAULT_DEVICE)
        log.info("Generating surface configuration from focal spot images")
        
        # Fit NURBS surface from images
        optimized_nurbs = self.fit_nurbs_from_images(scenario, heliostat_id, device)
        
        # Create FacetConfig following ARTIST patterns
        facet_config = FacetConfig(
            facet_key="image_learned_facet",
            control_points=optimized_nurbs.control_points.detach(),
            degree_e=optimized_nurbs.degree_e,
            degree_n=optimized_nurbs.degree_n,
            number_eval_points_e=self.number_eval_points_e,
            number_eval_points_n=self.number_eval_points_n,
            translation_vector=torch.zeros(4, device=device),  # No translation for single facet
            canting_e=torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device),
            canting_n=torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)
        )
        
        return [facet_config]
        
    def save_learned_surface(
        self, 
        surface_config: List[FacetConfig],
        output_path: pathlib.Path
    ) -> None:
        """
        Save learned surface configuration to file.
        
        Parameters
        ----------
        surface_config : List[FacetConfig]
            Surface configuration to save
        output_path : pathlib.Path
            Output file path
        """
        # Convert to serializable format
        config_data = {
            'surface_config': [],
            'converter_params': {
                'number_control_points_e': self.number_control_points_e,
                'number_control_points_n': self.number_control_points_n,
                'degree_e': self.degree_e,
                'degree_n': self.degree_n,
                'raytracing_loss_weight': self.raytracing_loss_weight,
                'image_loss_weight': self.image_loss_weight
            }
        }
        
        for facet in surface_config:
            facet_data = {
                'facet_key': facet.facet_key,
                'control_points': facet.control_points.cpu().numpy().tolist(),
                'degree_e': facet.degree_e,
                'degree_n': facet.degree_n,
                'number_eval_points_e': facet.number_eval_points_e,
                'number_eval_points_n': facet.number_eval_points_n,
                'translation_vector': facet.translation_vector.cpu().numpy().tolist(),
                'canting_e': facet.canting_e.cpu().numpy().tolist(),
                'canting_n': facet.canting_n.cpu().numpy().tolist()
            }
            config_data['surface_config'].append(facet_data)
            
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        log.info(f"Saved learned surface configuration to {output_path}")


def create_example_scenario_for_learning(device: Union[torch.device, str] = None) -> Scenario:
    """
    Create a minimal ARTIST scenario for surface learning demonstration.
    
    This creates the basic scenario structure needed for the image-based converter.
    """
    device = torch.device(device)
    
    # This is a simplified example - in practice you'd load from HDF5
    # For now, just demonstrate the converter pattern
    
    log.info("Creating example scenario for surface learning")
    log.info("In practice, load a real scenario with Scenario.load_scenario_from_hdf5()")
    
    # Return None for now - user should provide real scenario
    return None


# Example usage
if __name__ == "__main__":
    import torch
    
    logging.basicConfig(level=logging.INFO)
    device = get_default_device()
    
    # Example: Create converter with synthetic focal spot data
    converter = ImageBasedSurfaceConverter(
        raytracing_loss_weight=1.0,
        image_loss_weight=0.1,
        number_control_points_e=10,
        number_control_points_n=10,
        max_epoch=1000
    )
    
    # Add synthetic focal spot observations
    for i in range(5):
        # Synthetic image (replace with real camera data)
        image = torch.randn(100, 100) * 0.1 + torch.exp(
            -((torch.arange(100).float() - 50)**2 + 
              (torch.arange(100).float().view(-1, 1) - 50)**2) / 200
        )
        
        # Synthetic sun direction
        sun_dir = torch.tensor([0.1 * i, 0.0, -1.0])
        sun_dir = sun_dir / torch.norm(sun_dir)
        
        # Synthetic target position  
        target_pos = torch.tensor([0.0, 100.0, 5.0])
        
        converter.add_focal_spot_observation(image, sun_dir, target_pos)
        
    log.info(f"Added {len(converter.focal_spot_images)} focal spot observations")
    log.info("Ready for NURBS surface learning from real ARTIST scenario")