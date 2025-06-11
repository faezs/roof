#!/usr/bin/env python3
"""
Quick test for electrostatic mylar focal spot generation
"""

from test_surface_learning import DistortedMirrorGenerator, _run_single_test
import torch
import logging

# Reduce log noise
logging.getLogger().setLevel(logging.WARNING)

device = torch.device('mps')
mirror_gen = DistortedMirrorGenerator(device)
results = {}

# Test just one electrostatic pattern
voltage_pattern = torch.tensor([
    0, 50, 50, 0,
    50, 300, 300, 50, 
    50, 300, 300, 50,
    0, 50, 50, 0
], device=device, dtype=torch.float32)

print('Creating electrostatic surface...')
electrostatic_surface = mirror_gen.create_distorted_nurbs_surface(
    'electrostatic_mylar', 
    voltage_pattern=voltage_pattern
)

print('Running focal spot generation...')
_run_single_test('electrostatic_test', electrostatic_surface, mirror_gen, device, results)

print('Test completed!')
print(f'Results: {results}')