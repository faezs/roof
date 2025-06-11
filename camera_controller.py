"""
Raspberry Pi Camera System for Heliostat Field Monitoring

Hardware:
- Raspberry Pi 4B (8GB RAM)  
- HQ Camera Module (12.3MP Sony IMX477)
- 16mm C-mount lens (wide field view)
- Optional: FLIR Lepton thermal camera
"""

import asyncio
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import cv2
import torch
import zmq
import json
import logging
from pathlib import Path

# Pi Camera 2 for latest Raspberry Pi OS
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput, CircularOutput

# Image processing
from skimage import exposure, morphology, measure
from scipy import ndimage
import kornia  # PyTorch-based CV

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera system configuration"""
    # Camera settings
    resolution: Tuple[int, int] = (4056, 3040)  # 12.3MP
    framerate: int = 30
    exposure_mode: str = "auto"
    iso: int = 100
    
    # Processing settings
    hdr_enabled: bool = True
    denoise_enabled: bool = True
    
    # Network settings
    stream_port: int = 5555
    control_port: int = 5556
    
    # Storage settings
    buffer_size_mb: int = 512
    record_path: Path = Path("/home/pi/recordings")
    
    # Calibration
    calibration_file: Optional[Path] = None

class FocalSpotDetector:
    """Detect and analyze heliostat focal spots"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.calibration = self._load_calibration()
        
        # Pre-trained model for spot quality assessment
        self.quality_model = self._load_quality_model()
        
    def _load_calibration(self) -> Optional[Dict]:
        """Load camera calibration data"""
        if self.config.calibration_file and self.config.calibration_file.exists():
            return np.load(self.config.calibration_file)
        return None
        
    def _load_quality_model(self) -> torch.nn.Module:
        """Load pre-trained focal spot quality model"""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 3)  # [quality, centroid_x, centroid_y]
        )
        
        # Load weights if available
        model_path = Path("models/focal_spot_quality.pth")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
        return model
    
    def detect_spots(self, image: np.ndarray) -> List[Dict]:
        """Detect all focal spots in image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # HDR tone mapping if very bright spots
        if gray.max() > 250:
            gray = exposure.equalize_adapthist(gray)
            
        # Threshold to find bright spots
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small noise
                continue
                
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Extract spot region
            spot_img = gray[y:y+h, x:x+w]
            
            # Analyze spot quality
            quality_metrics = self.analyze_spot_quality(spot_img)
            
            spots.append({
                'heliostat_id': None,  # To be matched later
                'centroid': (cx, cy),
                'bounding_box': (x, y, w, h),
                'area': area,
                'intensity': np.mean(gray[y:y+h, x:x+w]),
                'quality': quality_metrics
            })
            
        return spots
    
    def analyze_spot_quality(self, spot_image: np.ndarray) -> Dict:
        """Analyze focal spot quality metrics"""
        # Resize to standard size
        spot_resized = cv2.resize(spot_image, (64, 64))
        
        # Convert to tensor
        spot_tensor = torch.tensor(spot_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spot_tensor = spot_tensor / 255.0
        
        # Run through quality model
        with torch.no_grad():
            outputs = self.quality_model(spot_tensor)
            quality_score = torch.sigmoid(outputs[0, 0]).item()
            
        # Classical metrics
        metrics = {
            'quality_score': quality_score,
            'peak_intensity': np.max(spot_image),
            'uniformity': np.std(spot_image) / (np.mean(spot_image) + 1e-6),
            'ellipticity': self._calculate_ellipticity(spot_image),
            'sharpness': self._calculate_sharpness(spot_image)
        }
        
        return metrics
    
    def _calculate_ellipticity(self, image: np.ndarray) -> float:
        """Calculate how elliptical vs circular the spot is"""
        # Find contour of spot
        _, binary = cv2.threshold(image, image.max() * 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Fit ellipse
            ellipse = cv2.fitEllipse(contours[0])
            (_, _), (MA, ma), _ = ellipse
            return 1.0 - (ma / MA if MA > 0 else 0)
        return 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate edge sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)

class SafetyMonitor:
    """Monitor field for safety hazards"""
    
    def __init__(self):
        # YOLO for person/animal detection
        self.detector = self._load_yolo()
        self.alert_callback = None
        
    def _load_yolo(self):
        """Load YOLO model for object detection"""
        # Using YOLOv8 nano for Pi
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        return model
    
    def check_frame(self, image: np.ndarray) -> List[Dict]:
        """Check frame for safety hazards"""
        # Run detection
        results = self.detector(image, stream=True, conf=0.5)
        
        hazards = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)
                # Check for people (0) or animals (various classes)
                if cls in [0, 15, 16, 17, 18, 19, 20]:  # person, cat, dog, horse, sheep, cow, bird
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    hazards.append({
                        'type': 'person' if cls == 0 else 'animal',
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf),
                        'timestamp': datetime.now()
                    })
                    
        return hazards

class ThermalCamera:
    """Optional FLIR Lepton thermal camera integration"""
    
    def __init__(self):
        try:
            from pylepton import Lepton
            self.lepton = Lepton()
            self.enabled = True
        except:
            logger.warning("Thermal camera not available")
            self.enabled = False
            
    def capture(self) -> Optional[np.ndarray]:
        """Capture thermal image"""
        if not self.enabled:
            return None
            
        with self.lepton as l:
            frame, _ = l.capture()
        
        # Convert to temperature in Celsius
        # Lepton outputs 14-bit values, scale to temperature
        temp_c = (frame / 100.0) - 273.15
        return temp_c
    
    def detect_hotspots(self, thermal_image: np.ndarray, threshold: float = 80.0) -> List[Dict]:
        """Detect dangerous hot spots"""
        if thermal_image is None:
            return []
            
        # Find regions above threshold
        hot_mask = thermal_image > threshold
        
        # Label connected regions
        labeled, num_features = ndimage.label(hot_mask)
        
        hotspots = []
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < 10:  # Filter small regions
                continue
                
            y_coords, x_coords = np.where(region)
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            max_temp = np.max(thermal_image[region])
            
            hotspots.append({
                'center': (center_x, center_y),
                'max_temperature': float(max_temp),
                'area_pixels': int(np.sum(region)),
                'timestamp': datetime.now()
            })
            
        return hotspots

class CameraController:
    """Main camera system controller"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = None
        self.spot_detector = FocalSpotDetector(config)
        self.safety_monitor = SafetyMonitor()
        self.thermal_camera = ThermalCamera()
        
        # ZMQ for communication
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{config.stream_port}")
        
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{config.control_port}")
        
        # Recording
        self.encoder = None
        self.circular_buffer = None
        
    def initialize_camera(self):
        """Initialize Pi Camera 2"""
        self.camera = Picamera2()
        
        # Configure for high quality capture
        config = self.camera.create_still_configuration(
            main={"size": self.config.resolution},
            raw={"size": self.config.resolution}
        )
        self.camera.configure(config)
        
        # Set camera controls
        self.camera.set_controls({
            "ExposureTime": 10000,  # 10ms
            "AnalogueGain": 1.0,
            "ColourGains": (2.0, 2.0)  # Neutral white balance
        })
        
        # Initialize circular buffer for pre-event recording
        self.circular_buffer = CircularOutput(
            buffersize=self.config.buffer_size_mb * 1024 * 1024
        )
        
        self.camera.start()
        logger.info("Camera initialized")
        
    async def capture_loop(self):
        """Main capture and processing loop"""
        frame_count = 0
        
        while True:
            # Capture frame
            frame = self.camera.capture_array("main")
            
            # HDR processing if enabled
            if self.config.hdr_enabled:
                frame = self.process_hdr(frame)
                
            # Detect focal spots
            spots = self.spot_detector.detect_spots(frame)
            
            # Safety check every 10 frames
            if frame_count % 10 == 0:
                hazards = self.safety_monitor.check_frame(frame)
                if hazards:
                    await self.handle_safety_alert(hazards)
                    
            # Thermal imaging if available
            if frame_count % 30 == 0:  # Every second
                thermal = self.thermal_camera.capture()
                if thermal is not None:
                    hotspots = self.thermal_camera.detect_hotspots(thermal)
                    if hotspots:
                        await self.handle_thermal_alert(hotspots)
                        
            # Prepare message
            message = {
                'timestamp': datetime.now().isoformat(),
                'frame_id': frame_count,
                'focal_spots': spots,
                'image_stats': {
                    'mean': float(np.mean(frame)),
                    'max': float(np.max(frame)),
                    'saturated_pixels': int(np.sum(frame > 250))
                }
            }
            
            # Publish results
            self.publisher.send_json(message)
            
            # Optionally send compressed frame
            if frame_count % 5 == 0:  # Every 5 frames
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.publisher.send(jpeg.tobytes(), flags=zmq.SNDMORE)
                
            frame_count += 1
            await asyncio.sleep(1.0 / self.config.framerate)
            
    def process_hdr(self, frame: np.ndarray) -> np.ndarray:
        """HDR tone mapping for high dynamic range scenes"""
        # Capture multiple exposures
        exposures = []
        exposure_times = [5000, 10000, 20000]  # microseconds
        
        for exp_time in exposure_times:
            self.camera.set_controls({"ExposureTime": exp_time})
            exposures.append(self.camera.capture_array("main"))
            
        # Merge exposures using Mertens fusion
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(exposures)
        
        # Convert back to 8-bit
        hdr_8bit = np.clip(hdr * 255, 0, 255).astype(np.uint8)
        
        return hdr_8bit
    
    async def handle_safety_alert(self, hazards: List[Dict]):
        """Handle safety hazard detection"""
        logger.warning(f"Safety hazard detected: {hazards}")
        
        # Save video buffer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.config.record_path / f"safety_alert_{timestamp}.mp4"
        
        # Dump circular buffer to file
        if self.circular_buffer:
            self.circular_buffer.fileoutput = str(filename)
            self.circular_buffer.start()
            await asyncio.sleep(10)  # Record 10 seconds after event
            self.circular_buffer.stop()
            
        # Send emergency stop command
        emergency_msg = {
            'command': 'EMERGENCY_STOP',
            'reason': 'safety_hazard',
            'hazards': hazards,
            'video_file': str(filename)
        }
        
        # Broadcast to all heliostats
        self.publisher.send_json(emergency_msg)
        
    async def handle_thermal_alert(self, hotspots: List[Dict]):
        """Handle thermal hotspot detection"""
        logger.warning(f"Thermal hotspot detected: {hotspots}")
        
        # Send defocus command to affected heliostats
        defocus_msg = {
            'command': 'DEFOCUS',
            'reason': 'thermal_hotspot',
            'hotspots': hotspots
        }
        
        self.publisher.send_json(defocus_msg)
        
    async def control_handler(self):
        """Handle control commands"""
        while True:
            try:
                message = await asyncio.get_event_loop().run_in_executor(
                    None, self.control_socket.recv_json
                )
                
                response = await self.process_command(message)
                self.control_socket.send_json(response)
                
            except Exception as e:
                logger.error(f"Control error: {e}")
                self.control_socket.send_json({'error': str(e)})
                
    async def process_command(self, command: Dict) -> Dict:
        """Process control commands"""
        cmd_type = command.get('type')
        
        if cmd_type == 'capture_calibration':
            # Capture high-res image for calibration
            frame = self.camera.capture_array("raw")
            spots = self.spot_detector.detect_spots(frame)
            
            # Save calibration data
            calib_data = {
                'timestamp': datetime.now().isoformat(),
                'spots': spots,
                'camera_settings': self.camera.camera_controls
            }
            
            filename = self.config.record_path / f"calibration_{datetime.now():%Y%m%d_%H%M%S}.npz"
            np.savez(filename, frame=frame, **calib_data)
            
            return {'status': 'success', 'filename': str(filename)}
            
        elif cmd_type == 'adjust_exposure':
            exposure = command.get('exposure_time', 10000)
            self.camera.set_controls({"ExposureTime": exposure})
            return {'status': 'success', 'exposure_time': exposure}
            
        elif cmd_type == 'start_recording':
            duration = command.get('duration', 60)
            filename = self.config.record_path / f"recording_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            
            encoder = H264Encoder(quality=Quality.HIGH)
            output = FfmpegOutput(str(filename))
            
            self.camera.start_encoder(encoder, output)
            await asyncio.sleep(duration)
            self.camera.stop_encoder()
            
            return {'status': 'success', 'filename': str(filename)}
            
        else:
            return {'status': 'error', 'message': f'Unknown command: {cmd_type}'}
    
    async def run(self):
        """Run the camera controller"""
        self.initialize_camera()
        
        # Start async tasks
        tasks = [
            asyncio.create_task(self.capture_loop()),
            asyncio.create_task(self.control_handler())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down camera controller")
        finally:
            self.camera.stop()
            self.context.destroy()

def main():
    """Main entry point"""
    config = CameraConfig()
    controller = CameraController(config)
    
    asyncio.run(controller.run())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()