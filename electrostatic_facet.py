"""
Electrostatic NURBS Facet for voltage-controlled mylar heliostat surfaces.

This module extends ARTIST's NurbsFacet to support dynamic surface deformation
based on electrostatic voltage patterns applied to a 4x4 electrode grid.
"""
from typing import Optional
import torch
from artist.field.facets_nurbs import NurbsFacet
from artist.util.nurbs import NURBSSurface


class ElectrostaticNurbsFacet(NurbsFacet):
    """
    A NURBS facet with electrostatic voltage control for mylar membranes.
    
    Extends ARTIST's NurbsFacet to dynamically modify control points based on
    voltage patterns applied to a 4x4 electrode grid, enabling real-time
    surface deformation modeling for flexible heliostat surfaces.
    
    Additional Attributes
    ----------
    voltage_pattern : torch.Tensor
        16-element tensor representing voltages for 4x4 electrode grid (0-300V)
    base_control_points : torch.Tensor
        Original flat surface control points (stored for reset/reference)
    electrode_gap : float
        Physical gap between electrodes and membrane (meters)
    max_voltage : float
        Maximum safe voltage (volts) 
    pull_in_voltage : float
        Voltage at which pull-in instability occurs
    
    Methods
    -------
    set_voltage_pattern(voltage_pattern)
        Update the voltage pattern and recompute control points
    _apply_electrostatic_deformation(control_points, voltage_pattern)
        Apply physics-based deformation to control points
    create_nurbs_surface(device)
        Create NURBS surface with current voltage-deformed control points
    """
    
    def __init__(
        self,
        control_points: torch.Tensor,
        degree_e: int,
        degree_n: int,
        number_eval_points_e: int,
        number_eval_points_n: int,
        translation_vector: torch.Tensor,
        canting_e: torch.Tensor,
        canting_n: torch.Tensor,
        voltage_pattern: Optional[torch.Tensor] = None,
        electrode_gap: float = 0.090,  # 90mm gap - realistic for heliostat scale electrostatic actuation
        max_voltage: float = 300.0,
        pull_in_voltage: float = 150.0,
    ) -> None:
        """
        Initialize an electrostatic NURBS facet.
        
        Parameters
        ----------
        voltage_pattern : Optional[torch.Tensor]
            16-element voltage pattern for 4x4 electrode grid (default: all zeros)
        electrode_gap : float
            Gap between electrodes and membrane in meters (default: 0.005)
        max_voltage : float
            Maximum safe voltage (default: 300.0)
        pull_in_voltage : float  
            Voltage threshold for pull-in instability (default: 150.0)
        """
        # Store base control points for deformation reference
        self.base_control_points = control_points.clone()
        
        # Initialize with flat surface first
        super().__init__(
            control_points=control_points,
            degree_e=degree_e,
            degree_n=degree_n,
            number_eval_points_e=number_eval_points_e,
            number_eval_points_n=number_eval_points_n,
            translation_vector=translation_vector,
            canting_e=canting_e,
            canting_n=canting_n,
        )
        
        # Electrostatic parameters
        self.electrode_gap = electrode_gap
        self.max_voltage = max_voltage
        self.pull_in_voltage = pull_in_voltage
        
        # Initialize voltage pattern
        if voltage_pattern is None:
            self.voltage_pattern = torch.zeros(16, device=control_points.device)
        else:
            self.voltage_pattern = voltage_pattern
            
        # Apply initial voltage deformation
        self._update_control_points()
    
    def set_voltage_pattern(self, voltage_pattern: torch.Tensor) -> None:
        """
        Update the voltage pattern and recompute control points.
        
        Parameters
        ----------
        voltage_pattern : torch.Tensor
            16-element tensor with voltages for 4x4 electrode grid
        """
        if voltage_pattern.shape[0] != 16:
            raise ValueError(f"Expected 16 voltages for 4x4 grid, got {voltage_pattern.shape[0]}")
            
        self.voltage_pattern = voltage_pattern.to(self.base_control_points.device)
        self._update_control_points()
    
    def _update_control_points(self) -> None:
        """Update control points based on current voltage pattern."""
        self.control_points = self._apply_electrostatic_deformation(
            self.base_control_points.clone(), self.voltage_pattern
        )
    
    def _apply_electrostatic_deformation(
        self, control_points: torch.Tensor, voltage_pattern: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply physics-based electrostatic deformation to NURBS control points.
        
        Models mylar membrane deflection under 4x4 electrode grid voltage pattern.
        Uses physics-inspired mapping from electrode voltages to control point positions.
        
        Args:
            control_points: [6, 6, 3] initial flat control points
            voltage_pattern: [16] voltages for 4x4 electrode grid (0-300V)
            
        Returns:
            Modified control points with electrostatic deflection
        """
        if voltage_pattern.numel() == 0:
            return control_points
            
        device = control_points.device
        
        # Reshape voltage pattern to 4x4 electrode grid
        electrode_voltages = voltage_pattern.reshape(4, 4)
        
        # Define electrode positions in membrane coordinates (normalized 0-1)
        electrode_positions = torch.zeros(4, 4, 2, device=device)
        for i in range(4):
            for j in range(4):
                electrode_positions[i, j, 0] = i / 3.0  # x position (0 to 1)
                electrode_positions[i, j, 1] = j / 3.0  # y position (0 to 1)
        
        # Control point grid dimensions
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
                        # Distance from control point to electrode (3D with physical gap)
                        dx = cp_x - electrode_positions[elec_i, elec_j, 0]
                        dy = cp_y - electrode_positions[elec_i, elec_j, 1]
                        # CRITICAL: Include physical gap to prevent infinite forces
                        distance = torch.sqrt(dx**2 + dy**2 + (self.electrode_gap)**2)
                        
                        # Electrode voltage
                        voltage = electrode_voltages[elec_i, elec_j]
                        
                        # SINGLE-SIDED ELECTROSTATIC PHYSICS: F = ε₀AVsig²/(2d²)
                        # Only one stator electrode behind membrane (no front electrode)
                        if voltage > 0:
                            # Physical constants
                            epsilon_0 = 8.854e-12  # F/m (permittivity of free space)
                            electrode_area = 0.01  # m² (10cm² electrode for heliostat scale)
                            
                            # Single-sided force: F = ε₀ * A * V² / (2 * d²)
                            force_per_area = epsilon_0 * electrode_area * (voltage**2) / (2 * self.electrode_gap**2)
                            
                            # Convert force to deflection using membrane compliance
                            # For thin mylar membrane: much more compliant than estimated
                            membrane_compliance = 1e3  # m³/N (much higher compliance for thin films)
                            base_deflection = force_per_area * membrane_compliance  # m
                            
                            # Spatial influence: electrode affects nearby control points more
                            # Use lateral distance only (not including gap)
                            lateral_distance = torch.sqrt(dx**2 + dy**2 + 1e-6)  # Avoid division by zero
                            influence_scale = 0.1  # 10cm influence radius
                            spatial_influence = torch.exp(-(lateral_distance / influence_scale)**2)
                            
                            # Final deflection = base deflection × spatial influence
                            electrode_deflection = -base_deflection * spatial_influence  # Attractive (negative Z)
                        else:
                            electrode_deflection = 0.0
                        
                        total_deflection += electrode_deflection
                
                # Apply boundary conditions (edges are more constrained)
                edge_factor = 1.0
                if cp_i == 0 or cp_i == num_ctrl_e-1 or cp_j == 0 or cp_j == num_ctrl_n-1:
                    edge_factor = 0.6  # Edge control points move less but not too constrained
                
                # Apply edge constraints
                total_deflection = total_deflection * edge_factor
                
                # Physical limit: membrane cannot deflect more than the gap
                max_safe_deflection = -self.electrode_gap * 0.8  # 80% of gap for safety margin
                total_deflection = torch.clamp(total_deflection, max_safe_deflection, 0.0)
                
                # Apply deflection to Z coordinate
                control_points[cp_i, cp_j, 2] += total_deflection
        
        return control_points

    def create_nurbs_surface(
        self, device: Optional[torch.device] = None
    ) -> NURBSSurface:
        """
        Create a NURBS surface using current voltage-deformed control points.
        
        This overrides the parent method to ensure the most current control points
        (reflecting the current voltage pattern) are used for surface generation.
        
        Parameters
        ----------
        device : Optional[torch.device]
            The device on which to perform computations (default is None)
            
        Returns
        -------
        NURBSSurface
            The NURBS surface with current electrostatic deformation
        """
        # Ensure control points are current with voltage pattern
        self._update_control_points()
        
        # Use parent method with updated control points
        return super().create_nurbs_surface(device=device)