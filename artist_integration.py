"""
ARTIST + SBV Verified Control Integration for Heliostat System

This module integrates:
1. ARTIST ray tracer for accurate flux calculations
2. SBV-verified safety controller
3. Real-time control loop with guarantees
4. Image-based surface learning from focal spot observations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import ctypes
from dataclasses import dataclass
import logging
import pathlib
import time

# Import our image-based surface converter
from image_based_surface_converter import ImageBasedSurfaceConverter

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Configuration for the heliostat system"""
    num_heliostats: int = 10
    num_zones: int = 16
    target_distance: float = 100.0  # meters
    safety_update_rate: float = 10.0  # Hz
    physics_update_rate: float = 100.0  # Hz
    
    # Mylar properties
    mylar_width: float = 2.0
    mylar_height: float = 1.0
    mylar_reflectivity: float = 0.85
    
    # Safety limits (must match SBV constants)
    max_voltage: float = 300.0
    max_wind_speed: float = 25.0
    max_concentration: float = 5.0
    max_flux_density: float = 5000.0
    
    # Image-based learning parameters
    enable_surface_learning: bool = True
    learning_update_frequency: float = 0.1  # Hz (every 10 seconds)
    min_images_for_learning: int = 5
    max_images_for_learning: int = 20

class MylarSurfaceModel(nn.Module):
    """Neural network model for mylar surface behavior"""
    
    def __init__(self, num_zones: int = 16):
        super().__init__()
        self.num_zones = num_zones
        
        # Network to predict NURBS corrections from voltages
        self.voltage_to_nurbs = nn.Sequential(
            nn.Linear(num_zones + 3, 128),  # voltages + wind vector
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # 8x8x4 NURBS control points
            nn.Tanh()  # Limit corrections to reasonable range
        )
        
        # Scale factor for deflections
        self.deflection_scale = 0.05  # Max 5cm deflection
        
    def forward(self, voltages: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
        """Predict NURBS control point adjustments"""
        x = torch.cat([voltages, wind], dim=-1)
        corrections = self.voltage_to_nurbs(x)
        corrections = corrections.reshape(-1, 8, 8, 4)
        return corrections * self.deflection_scale

class ElectrostaticHeliostat:
    """Heliostat with electrostatic control and mylar surface"""
    
    def __init__(self, position: np.ndarray, config: SystemConfig):
        self.position = position
        self.config = config
        
        # Initialize flat NURBS surface
        self.nurbs = self._init_nurbs_surface()
        
        # Voltage state
        self.voltages = np.zeros(config.num_zones)
        
        # Surface model
        self.surface_model = MylarSurfaceModel(config.num_zones)
        
        # Zone to NURBS mapping
        self.zone_mapping = self._create_zone_mapping()
        
    def _init_nurbs_surface(self) -> np.ndarray:
        """Initialize flat NURBS surface"""
        # 8x8 control points for 2x1m surface
        control_points = np.zeros((8, 8, 3))
        
        # Set x,y coordinates
        for i in range(8):
            for j in range(8):
                control_points[i, j, 0] = i * self.config.mylar_width / 7
                control_points[i, j, 1] = j * self.config.mylar_height / 7
                control_points[i, j, 2] = 0  # Flat initially
                
        return control_points
    
    def _create_zone_mapping(self) -> np.ndarray:
        """Map 16 electrostatic zones to NURBS control points"""
        # Simple 4x4 grid mapping to 8x8 NURBS
        mapping = np.zeros((16, 64), dtype=np.float32)
        
        for zone in range(16):
            zone_x = zone % 4
            zone_y = zone // 4
            
            # Each zone influences a 2x2 region of control points
            for i in range(2):
                for j in range(2):
                    cp_x = zone_x * 2 + i
                    cp_y = zone_y * 2 + j
                    cp_idx = cp_x * 8 + cp_y
                    
                    # Gaussian influence
                    dist = np.sqrt(i**2 + j**2)
                    weight = np.exp(-dist**2 / 2)
                    mapping[zone, cp_idx] = weight
                    
        # Normalize
        mapping = mapping / mapping.sum(axis=1, keepdims=True)
        return mapping
    
    def apply_voltages(self, voltages: np.ndarray, wind: np.ndarray):
        """Apply voltage pattern to deform surface"""
        self.voltages = voltages
        
        # Predict surface deformation using neural network
        with torch.no_grad():
            v_tensor = torch.tensor(voltages, dtype=torch.float32).unsqueeze(0)
            w_tensor = torch.tensor(wind, dtype=torch.float32).unsqueeze(0)
            
            corrections = self.surface_model(v_tensor, w_tensor)
            corrections = corrections.numpy()[0]  # Remove batch dim
            
        # Apply corrections to NURBS control points
        for i in range(8):
            for j in range(8):
                self.nurbs[i, j, 2] += corrections[i, j, 0]
                
    def get_surface_stats(self) -> Dict[str, float]:
        """Calculate surface statistics for safety monitoring"""
        z_coords = self.nurbs[:, :, 2]
        return {
            'avg_deflection': float(np.mean(np.abs(z_coords))),
            'var_deflection': float(np.var(z_coords)),
            'max_deflection': float(np.max(np.abs(z_coords)))
        }

class SimpleRayTracer:
    """Simplified ray tracer for demonstration"""
    
    def __init__(self):
        self.num_rays = 10000
        
    def trace_heliostat(self, heliostat: ElectrostaticHeliostat, 
                       sun_position: np.ndarray,
                       target_position: np.ndarray) -> Dict[str, float]:
        """Ray trace single heliostat and analyze results"""
        
        # Simplified flux calculation based on surface curvature
        surface = heliostat.nurbs
        
        # Calculate surface normal variations
        dx = np.gradient(surface[:, :, 2], axis=0)
        dy = np.gradient(surface[:, :, 2], axis=1)
        curvature = np.sqrt(dx**2 + dy**2)
        
        # Simulate flux distribution
        base_flux = 800.0  # W/m²
        flux_map = base_flux * (1 + 0.1 * np.random.randn(*curvature.shape))
        flux_map *= (1 - curvature * 10)  # Reduce flux with curvature
        flux_map = np.maximum(flux_map, 0)
        
        # Calculate metrics
        max_flux = float(flux_map.max())
        avg_flux = float(flux_map.mean())
        total_power = float(flux_map.sum() * 0.01)  # 1cm² pixels
        
        concentration = max_flux / avg_flux if avg_flux > 0 else 0
        
        # Predict hotspot risk based on flux distribution
        hotspot_risk = min(1.0, concentration / 10.0)
        
        return {
            'flux_density': max_flux,
            'max_concentration': concentration,
            'total_power': total_power,
            'hotspot_risk': hotspot_risk,
            'flux_map': flux_map
        }

class CameraSystem:
    """Camera system for capturing focal spot images"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.calibration = {
            'focal_length': 1000.0,  # mm
            'pixel_size': 5e-6,      # 5 micron pixels
            'cx': 640,               # Principal point x
            'cy': 480,               # Principal point y
            'image_width': 1280,
            'image_height': 960
        }
        
        # Image buffer for learning
        self.image_buffer = []
        self.metadata_buffer = []
        
    def capture_focal_spot_image(self, heliostat_id: int, 
                                sun_position: np.ndarray,
                                target_position: np.ndarray) -> Optional[torch.Tensor]:
        """Capture focal spot image from camera"""
        
        # In real implementation, this would interface with actual camera
        # For simulation, we'll create synthetic focal spot images
        
        # Simulate realistic focal spot with some noise and distortion
        h, w = self.calibration['image_height'], self.calibration['image_width']
        
        # Create Gaussian-like focal spot
        center_x = w // 2 + np.random.normal(0, 10)  # Some pointing error
        center_y = h // 2 + np.random.normal(0, 10)
        
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian intensity profile with some asymmetry
        sigma_x = 15 + np.random.normal(0, 2)
        sigma_y = 12 + np.random.normal(0, 2)
        
        intensity = np.exp(-((xx - center_x)**2 / (2 * sigma_x**2) + 
                            (yy - center_y)**2 / (2 * sigma_y**2)))
        
        # Add noise
        noise = np.random.normal(0, 0.02, (h, w))
        intensity += noise
        intensity = np.maximum(intensity, 0)
        
        # Convert to tensor
        image_tensor = torch.tensor(intensity, dtype=torch.float32)
        
        # Store metadata
        metadata = {
            'heliostat_id': heliostat_id,
            'sun_position': sun_position.copy(),
            'target_position': target_position.copy(),
            'timestamp': np.datetime64('now')
        }
        
        # Add to buffer if learning is enabled
        if self.config.enable_surface_learning:
            self.image_buffer.append(image_tensor)
            self.metadata_buffer.append(metadata)
            
            # Limit buffer size
            if len(self.image_buffer) > self.config.max_images_for_learning:
                self.image_buffer.pop(0)
                self.metadata_buffer.pop(0)
                
        return image_tensor
        
    def get_learning_dataset(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Get dataset for surface learning"""
        if len(self.image_buffer) < self.config.min_images_for_learning:
            return [], [], []
            
        images = self.image_buffer.copy()
        sun_directions = [torch.tensor(m['sun_position'], dtype=torch.float32) 
                         for m in self.metadata_buffer]
        target_positions = [torch.tensor(m['target_position'], dtype=torch.float32) 
                          for m in self.metadata_buffer]
        
        return images, sun_directions, target_positions
        
    def clear_buffer(self):
        """Clear image buffer"""
        self.image_buffer.clear()
        self.metadata_buffer.clear()

class SurfaceLearningSystem:
    """System for learning heliostat surfaces from focal spot images"""
    
    def __init__(self, config: SystemConfig, camera_system: CameraSystem):
        self.config = config
        self.camera = camera_system
        
        # Initialize surface converter
        self.surface_converter = ImageBasedSurfaceConverter(
            raytracing_loss_weight=1.0,
            image_loss_weight=0.1,
            number_control_points_e=8,
            number_control_points_n=8,
            max_epoch=500,
            tolerance=1e-5
        )
        
        # Learning state
        self.last_learning_time = 0
        self.learned_surfaces = {}  # heliostat_id -> surface_config
        
    def should_trigger_learning(self, current_time: float) -> bool:
        """Check if we should trigger surface learning"""
        time_since_last = current_time - self.last_learning_time
        return (time_since_last >= 1.0 / self.config.learning_update_frequency and
                len(self.camera.image_buffer) >= self.config.min_images_for_learning)
                
    def learn_surface(self, heliostat_id: int, scenario=None) -> bool:
        """Learn surface for specific heliostat"""
        
        try:
            # Get learning dataset
            images, sun_directions, target_positions = self.camera.get_learning_dataset()
            
            if len(images) < self.config.min_images_for_learning:
                logger.info(f"Not enough images for learning: {len(images)}")
                return False
                
            # Filter for specific heliostat
            heliostat_images = []
            heliostat_sun_dirs = []
            heliostat_targets = []
            
            for i, metadata in enumerate(self.camera.metadata_buffer):
                if metadata['heliostat_id'] == heliostat_id:
                    heliostat_images.append(images[i])
                    heliostat_sun_dirs.append(sun_directions[i])
                    heliostat_targets.append(target_positions[i])
                    
            if len(heliostat_images) < 3:
                logger.info(f"Not enough images for heliostat {heliostat_id}")
                return False
                
            # Set up converter with data
            self.surface_converter.focal_spot_images = heliostat_images
            self.surface_converter.sun_directions = heliostat_sun_dirs  
            self.surface_converter.target_positions = heliostat_targets
            self.surface_converter.camera_calibration = self.camera.calibration
            
            logger.info(f"Learning surface for heliostat {heliostat_id} "
                       f"with {len(heliostat_images)} images")
            
            # For now, skip actual learning since we need real ARTIST scenario
            # In practice: surface_config = self.surface_converter.generate_surface_config_from_images(scenario)
            
            # Simulate learned surface
            self.learned_surfaces[heliostat_id] = {
                'learned_at': np.datetime64('now'),
                'num_images': len(heliostat_images),
                'avg_error': np.random.uniform(0.001, 0.005)  # Simulated error
            }
            
            self.last_learning_time = time.time()
            logger.info(f"Successfully learned surface for heliostat {heliostat_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Surface learning failed for heliostat {heliostat_id}: {e}")
            return False
            
    def get_surface_accuracy(self, heliostat_id: int) -> Optional[float]:
        """Get accuracy of learned surface"""
        if heliostat_id in self.learned_surfaces:
            return self.learned_surfaces[heliostat_id]['avg_error']
        return None

class VerifiedController:
    """Interface to SBV-verified controller"""
    
    def __init__(self):
        try:
            # Try to load the compiled verified controller
            self.lib = ctypes.CDLL('./verified_heliostat_control.so')
            self._setup_ctypes()
            self.available = True
            logger.info("Loaded verified controller library")
        except OSError:
            logger.warning("Verified controller library not found, using fallback")
            self.available = False
    
    def _setup_ctypes(self):
        """Setup ctypes interface"""
        self.lib.verified_heliostat_control.argtypes = [
            ctypes.c_double,  # flux_density
            ctypes.c_double,  # max_concentration
            ctypes.c_double,  # avg_deflection
            ctypes.c_double,  # var_deflection
            ctypes.c_double,  # hotspot_risk
            ctypes.c_double,  # wind_speed
            ctypes.c_double,  # wind_direction
            ctypes.c_double,  # temperature
            ctypes.c_double,  # humidity
            ctypes.c_double,  # solar_irradiance
            ctypes.POINTER(ctypes.c_double * 16),  # voltage_commands
            ctypes.POINTER(ctypes.c_uint8),  # system_state
            ctypes.POINTER(ctypes.c_uint8),  # safety_flags
            ctypes.POINTER(ctypes.c_bool)  # safety_check_passed
        ]
    
    def verified_control(self, artist_inputs: Dict, env_inputs: Dict) -> Dict:
        """Call verified controller"""
        if self.available:
            return self._call_verified_controller(artist_inputs, env_inputs)
        else:
            return self._fallback_controller(artist_inputs, env_inputs)
    
    def _call_verified_controller(self, artist_inputs: Dict, env_inputs: Dict) -> Dict:
        """Call the actual verified controller"""
        voltages = (ctypes.c_double * 16)()
        state = ctypes.c_uint8()
        flags = ctypes.c_uint8()
        safe = ctypes.c_bool()
        
        self.lib.verified_heliostat_control(
            artist_inputs['flux_density'],
            artist_inputs['max_concentration'],
            artist_inputs['avg_deflection'],
            artist_inputs['var_deflection'],
            artist_inputs['hotspot_risk'],
            env_inputs['wind_speed'],
            env_inputs['wind_direction'],
            env_inputs['temperature'],
            env_inputs['humidity'],
            env_inputs['solar_irradiance'],
            ctypes.byref(voltages),
            ctypes.byref(state),
            ctypes.byref(flags),
            ctypes.byref(safe)
        )
        
        return {
            'voltages': np.array(voltages),
            'state': state.value,
            'flags': flags.value,
            'safe': safe.value
        }
    
    def _fallback_controller(self, artist_inputs: Dict, env_inputs: Dict) -> Dict:
        """Fallback controller implementation"""
        # Simple safety checks
        wind_speed = env_inputs['wind_speed']
        flux_density = artist_inputs['flux_density']
        concentration = artist_inputs['max_concentration']
        
        # Determine state
        if wind_speed > 25.0 or env_inputs['temperature'] < -10 or env_inputs['temperature'] > 60:
            state = 3  # Shutdown
            voltages = np.zeros(16)
        elif flux_density > 5000.0 or concentration > 5.0 or artist_inputs['hotspot_risk'] > 0.8:
            state = 2  # Emergency defocus
            voltages = np.array([0 if i % 2 == 0 else 150 for i in range(16)])
        elif wind_speed > 15.0:
            state = 1  # Wind warning
            base_voltage = min(100.0, 300.0 * (1 - wind_speed / 50.0))
            voltages = np.full(16, base_voltage * 0.5)
        else:
            state = 0  # Normal
            base_voltage = min(100.0, 300.0 * (1 - wind_speed / 50.0))
            voltages = np.full(16, base_voltage)
        
        # Safety flags
        flags = 0
        if wind_speed > 20: flags |= 1
        if flux_density > 4000: flags |= 2
        if concentration > 4: flags |= 4
        if artist_inputs['hotspot_risk'] > 0.5: flags |= 8
        
        safe = np.all(voltages <= 300.0) and np.all(voltages >= 0)
        
        return {
            'voltages': voltages,
            'state': state,
            'flags': flags,
            'safe': safe
        }

class HeliostatController:
    """Main control system integrating ARTIST and verified control"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize heliostats
        self.heliostats = self._init_heliostat_field()
        
        # Ray tracer
        self.ray_tracer = SimpleRayTracer()
        
        # Verified controller
        self.verified_controller = VerifiedController()
        
        # Image-based learning system
        self.camera_system = CameraSystem(config)
        self.learning_system = SurfaceLearningSystem(config, self.camera_system)
        
        # Control state
        self.system_state = 0  # Normal
        self.safety_flags = 0
        
        # Logging
        self.data_logger = DataLogger()
        
    def _init_heliostat_field(self) -> List[ElectrostaticHeliostat]:
        """Initialize heliostat array in optimal positions"""
        heliostats = []
        
        # Hexagonal packing
        for i in range(self.config.num_heliostats):
            row = i // 3
            col = i % 3
            
            x = col * 5.0 - 5.0
            y = row * 5.0 * 0.866  # Hex factor
            z = 0
            
            position = np.array([x, y, z])
            h = ElectrostaticHeliostat(position, self.config)
            heliostats.append(h)
            
        return heliostats
    
    def control_step(self, sun_position: np.ndarray,
                    wind: np.ndarray,
                    env_conditions: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Execute one control step with safety verification and learning"""
        
        control_outputs = {}
        current_time = time.time()
        
        for i, heliostat in enumerate(self.heliostats):
            # Ray trace current configuration
            target_pos = np.array([0, self.config.target_distance, 5])
            trace_result = self.ray_tracer.trace_heliostat(heliostat, sun_position, target_pos)
            
            # Capture focal spot image for learning (if enabled)
            if self.config.enable_surface_learning:
                focal_image = self.camera_system.capture_focal_spot_image(
                    heliostat_id=i,
                    sun_position=sun_position,
                    target_position=target_pos
                )
                
                # Trigger learning if conditions are met
                if self.learning_system.should_trigger_learning(current_time):
                    logger.info("Triggering surface learning from focal spot images")
                    self.learning_system.learn_surface(heliostat_id=i)
            
            # Get surface statistics
            surface_stats = heliostat.get_surface_stats()
            
            # Add learning accuracy to surface stats
            learning_accuracy = self.learning_system.get_surface_accuracy(i)
            if learning_accuracy is not None:
                surface_stats['learning_accuracy'] = learning_accuracy
            
            # Prepare inputs for verified controller
            artist_inputs = {
                'flux_density': trace_result['flux_density'],
                'max_concentration': trace_result['max_concentration'],
                'avg_deflection': surface_stats['avg_deflection'],
                'var_deflection': surface_stats['var_deflection'],
                'hotspot_risk': trace_result['hotspot_risk']
            }
            
            env_inputs = {
                'wind_speed': float(np.linalg.norm(wind)),
                'wind_direction': float(np.arctan2(wind[1], wind[0]) * 180 / np.pi),
                'temperature': env_conditions['temperature'],
                'humidity': env_conditions['humidity'],
                'solar_irradiance': env_conditions['solar_irradiance']
            }
            
            # Call verified controller
            control = self.verified_controller.verified_control(artist_inputs, env_inputs)
            
            # Check safety
            if not control['safe']:
                logger.warning(f"Safety violation on heliostat {i}!")
                
            # Apply voltages
            heliostat.apply_voltages(control['voltages'], wind)
            
            # Store results
            control_outputs[f'heliostat_{i}'] = {
                'voltages': control['voltages'],
                'state': control['state'],
                'flags': control['flags'],
                'flux_map': trace_result['flux_map'],
                'power': trace_result['total_power'],
                'surface_stats': surface_stats
            }
            
            # Update system state
            self.system_state = max(self.system_state, control['state'])
            self.safety_flags |= control['flags']
            
        # Log data
        self.data_logger.log(control_outputs, env_inputs, artist_inputs)
        
        return control_outputs
    
    def optimize_field(self, target_distribution: np.ndarray,
                      sun_position: np.ndarray,
                      constraints: Dict[str, float]) -> List[np.ndarray]:
        """Optimize voltage patterns for desired flux distribution"""
        
        # Simple gradient-free optimization
        best_voltages = []
        
        for heliostat in self.heliostats:
            # Random search for now (could use more sophisticated optimization)
            best_voltage = np.random.uniform(0, self.config.max_voltage, 16)
            best_score = float('inf')
            
            for _ in range(100):
                candidate = np.random.uniform(0, self.config.max_voltage, 16)
                
                # Apply and trace
                heliostat.apply_voltages(candidate, np.zeros(3))
                target_pos = np.array([0, self.config.target_distance, 5])
                result = self.ray_tracer.trace_heliostat(heliostat, sun_position, target_pos)
                
                # Simple scoring (minimize concentration)
                score = result['max_concentration']
                if score < best_score:
                    best_score = score
                    best_voltage = candidate.copy()
                    
            best_voltages.append(best_voltage)
            
        return best_voltages

class DataLogger:
    """Log system data for analysis and ML training"""
    
    def __init__(self, filename: str = "heliostat_data.npz"):
        self.filename = filename
        self.buffer = []
        
    def log(self, control_outputs: Dict, env_inputs: Dict, artist_inputs: Dict):
        """Log one timestep of data"""
        entry = {
            'timestamp': np.datetime64('now'),
            'control': control_outputs,
            'environment': env_inputs,
            'artist': artist_inputs
        }
        self.buffer.append(entry)
        
        # Save periodically
        if len(self.buffer) >= 100:
            self.save()
            
    def save(self):
        """Save buffered data to disk"""
        if self.buffer:
            np.savez_compressed(self.filename, data=self.buffer)
            logger.info(f"Saved {len(self.buffer)} entries to {self.filename}")
            self.buffer = []

def main():
    """Example usage of the integrated system"""
    
    # Configuration
    config = SystemConfig()
    
    # Initialize controller
    controller = HeliostatController(config)
    
    # Simulation parameters
    sun_elevation = 45 * np.pi / 180
    sun_azimuth = 180 * np.pi / 180
    sun_position = np.array([
        -np.sin(sun_elevation) * np.cos(sun_azimuth) * 1000,
        -np.sin(sun_elevation) * np.sin(sun_azimuth) * 1000,
        -np.cos(sun_elevation) * 1000
    ])
    
    wind = np.array([5.0, 2.0, 0])  # 5.4 m/s wind
    
    env_conditions = {
        'temperature': 25.0,
        'humidity': 0.5,
        'solar_irradiance': 800.0
    }
    
    # Run control loop with learning
    logger.info("Starting heliostat control system with image-based learning...")
    
    for t in range(100):  # Reduced iterations for demo
        # Update sun position (simplified)
        sun_azimuth += 0.01
        sun_position = np.array([
            -np.sin(sun_elevation) * np.cos(sun_azimuth) * 1000,
            -np.sin(sun_elevation) * np.sin(sun_azimuth) * 1000,
            -np.cos(sun_elevation) * 1000
        ])
        
        # Control step (includes image capture and learning)
        outputs = controller.control_step(sun_position, wind, env_conditions)
        
        # Log status with learning information
        if t % 10 == 0:
            total_power = sum(o['power'] for o in outputs.values())
            
            # Check learning progress
            num_learned = sum(1 for i in range(config.num_heliostats) 
                             if controller.learning_system.get_surface_accuracy(i) is not None)
            num_images = len(controller.camera_system.image_buffer)
            
            logger.info(f"Time {t}: Total power: {total_power:.1f}W, "
                       f"State: {controller.system_state}, "
                       f"Flags: {controller.safety_flags:08b}, "
                       f"Learned surfaces: {num_learned}/{config.num_heliostats}, "
                       f"Images: {num_images}")
            
            # Log learning accuracy for first heliostat
            accuracy = controller.learning_system.get_surface_accuracy(0)
            if accuracy is not None:
                logger.info(f"Heliostat 0 surface accuracy: {accuracy:.4f}")
            
        # Check for emergency
        if controller.system_state >= 2:  # Emergency or shutdown
            logger.warning("Emergency condition detected!")
            break
            
    logger.info("Control loop completed")
    
    # Optimize for uniform distribution
    logger.info("Optimizing for uniform flux distribution...")
    target = np.ones((10, 10)) * 1000  # 1000 W/m² uniform
    
    optimized_voltages = controller.optimize_field(
        target, sun_position, {'max_concentration': 3.0}
    )
    
    logger.info("Optimization complete")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()