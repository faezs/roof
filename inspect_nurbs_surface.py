#!/usr/bin/env python3
"""
Inspect NURBS surface deformation from voltage patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from test_surface_learning import DistortedMirrorGenerator

device = torch.device('mps')
mirror_gen = DistortedMirrorGenerator(device)

# Test different voltage patterns
voltage_patterns = {
    "flat": torch.zeros(16, device=device, dtype=torch.float32),
    
    "center_focus": torch.tensor([
        0, 50, 50, 0,
        50, 300, 300, 50, 
        50, 300, 300, 50,
        0, 50, 50, 0
    ], device=device, dtype=torch.float32),
    
    "edge_focus": torch.tensor([
        300, 200, 200, 300,
        200, 0, 0, 200,
        200, 0, 0, 200,
        300, 200, 200, 300
    ], device=device, dtype=torch.float32),
    
    "diagonal": torch.tensor([
        300, 100, 100, 0,
        100, 200, 150, 100,
        100, 150, 200, 100,
        0, 100, 100, 300
    ], device=device, dtype=torch.float32)
}

print("Creating surfaces and inspecting deformation...")

results = {}
for pattern_name, voltage_pattern in voltage_patterns.items():
    print(f"\n=== {pattern_name.upper()} PATTERN ===")
    
    if pattern_name == "flat":
        surface = mirror_gen.create_distorted_nurbs_surface('flat')
    else:
        surface = mirror_gen.create_distorted_nurbs_surface(
            'electrostatic_mylar', 
            voltage_pattern=voltage_pattern
        )
    
    # Get surface points and analyze them
    surface_points, surface_normals = surface.get_surface_points_and_normals(device=device)
    
    # Extract Z coordinates (height) - detach gradients first
    z_coords = surface_points[0, :, 2].detach().cpu().numpy()  # First facet, all points, Z coordinate
    
    print(f"Z coordinate range: [{z_coords.min():.6f}, {z_coords.max():.6f}]")
    print(f"Z coordinate std dev: {z_coords.std():.6f}")
    print(f"Max deflection: {abs(z_coords.min()):.6f}m = {abs(z_coords.min())*1000:.2f}mm")
    
    # Reshape Z coordinates to grid for visualization
    grid_size = int(np.sqrt(len(z_coords)))
    if grid_size * grid_size == len(z_coords):
        z_grid = z_coords.reshape(grid_size, grid_size)
        print(f"Reshaped to {grid_size}x{grid_size} grid")
        
        # Plot the surface deformation
        plt.figure(figsize=(6, 5))
        plt.imshow(z_grid, cmap='RdBu_r', interpolation='bilinear')
        plt.colorbar(label='Z deflection (m)')
        plt.title(f'Surface Deformation - {pattern_name.replace("_", " ").title()}')
        plt.xlabel('East direction')
        plt.ylabel('North direction')
        
        # Save the plot
        plt.savefig(f'surface_deformation_{pattern_name}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        results[pattern_name] = {
            'z_range': (z_coords.min(), z_coords.max()),
            'z_std': z_coords.std(),
            'max_deflection_mm': abs(z_coords.min()) * 1000,
            'grid_shape': z_grid.shape
        }
    else:
        print(f"Cannot reshape {len(z_coords)} points to square grid")
        results[pattern_name] = {
            'z_range': (z_coords.min(), z_coords.max()),
            'z_std': z_coords.std(),
            'max_deflection_mm': abs(z_coords.min()) * 1000,
            'grid_shape': 'irregular'
        }

print(f"\n=== SUMMARY ===")
for pattern_name, data in results.items():
    print(f"{pattern_name}: {data['max_deflection_mm']:.2f}mm deflection, std={data['z_std']:.6f}")

print(f"\nSurface deformation plots saved as 'surface_deformation_*.png'")