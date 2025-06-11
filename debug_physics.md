# Electrostatic Heliostat Physics Debug Context

## Current State
Working on proper electrostatic physics implementation for voltage-controlled mylar heliostat surfaces in ARTIST raytracing system.

## Problem Summary
- Started with fantasy physics producing 735+ meter deflections but correct optical effects
- User provided Sanders Electrostatic Loudspeaker Design Cookbook for proper physics
- Implemented proper electrostatic force equation: getting 4mm deflections with 90mm gap
- **CRITICAL ISSUE**: Focal spots are identical between flat and electrostatic patterns - 4mm deflection insufficient for optical control

## Key Physics Equations (from Sanders Cookbook)
From `/Users/faezs/roof/electrostatic_physics.txt` line 120:
**Fsig = ε₀AVpolVsig / d²**

Where:
- ε₀ = 8.854 × 10⁻¹² F/m (permittivity of free space)  
- A = electrode area (m²)
- Vpol = polarizing voltage (V)
- Vsig = signal voltage (V)
- d = gap distance (m)

For single-sided electrode (our case): **F = ε₀AV²/(2d²)**

## Current Implementation Status
In `electrostatic_facet.py:_apply_electrostatic_deformation()`:

### Physical Parameters (Current):
- `electrode_gap: float = 0.090` # 90mm gap - heliostat scale
- `epsilon_0 = 8.854e-12` # F/m (permittivity of free space)
- `electrode_area = 0.01` # m² (10cm² electrode)
- Single-sided force: `F = ε₀ * A * V² / (2 * d²)`
- `membrane_compliance = 1e3` # m³/N (thin mylar)
- `max_safe_deflection = -electrode_gap * 0.8` # 80% of gap limit = 72mm max
- **ACTUAL DEFLECTIONS**: 4mm (hitting some limit, not the 72mm clamp)

### Safety Constraints:
- Physical limit: membrane cannot deflect more than gap distance
- Current max deflection: ~72mm (80% of 90mm gap)
- Clamping: `torch.clamp(total_deflection, max_safe_deflection, 0.0)`

## Test Results History
1. **Fantasy Physics**: deflection_scale = 5e-11 * V²/gap² → 735m deflections, **good optical effects**
2. **First Real Physics**: Got 0.00mm deflections (too weak)
3. **Increased Compliance**: membrane_compliance = 1e3 → Still 0.00mm  
4. **Single-sided + 90mm gap**: 4mm deflections, **NO optical effects** (focal spots identical)

## CRITICAL FINDINGS
**Latest Test Results** (inspect_nurbs_surface.py):
```
=== FLAT PATTERN ===
Max deflection: 0.000000m = 0.00mm

=== CENTER_FOCUS PATTERN ===  
Max deflection: 0.004000m = 4.00mm

=== EDGE_FOCUS PATTERN ===
Max deflection: 0.004000m = 4.00mm

=== DIAGONAL PATTERN ===
Max deflection: 0.004000m = 4.00mm
```

**Focal Spot Analysis**: 
- focal_spots_flat.png: Normal focused spots
- focal_spots_electrostatic_mylar_strong_center_focus.png: **IDENTICAL to flat**
- All electrostatic patterns produce identical focal spots to flat surface

**ROOT CAUSE**: 4mm deflection over heliostat-scale surface produces negligible optical effects. Need much larger deflections or different approach.

## Key Files
- `electrostatic_facet.py`: ElectrostaticNurbsFacet class with physics model
- `test_surface_learning.py`: Test script for focal spot analysis
- `inspect_nurbs_surface.py`: Surface deformation analysis
- `electrostatic_physics.txt`: Sanders cookbook physics reference

## Voltage Patterns Being Tested
- FLAT: all zeros → 0mm deflection
- CENTER_FOCUS: higher voltage in center 4 electrodes → 4mm deflection  
- EDGE_FOCUS: higher voltage on edge electrodes → 4mm deflection
- DIAGONAL: diagonal voltage pattern → 4mm deflection

## Next Steps Needed
1. **Increase membrane compliance dramatically** - need cm-scale deflections (20-50mm) not mm-scale
2. **Alternative**: Reduce electrode gap to increase force (but check physical realism)
3. **Alternative**: Increase electrode area or voltage significantly
4. **Alternative**: Wire mesh electrode model with higher effective area
5. **Verify**: Check if 4mm is actually the deflection or if there's a calculation error

## Architecture
- ARTIST raytracing with custom ElectrostaticNurbsFacet extending NurbsFacet
- 4×4 electrode grid (16 voltages) controlling 6×6 NURBS control points
- Spatial influence function with exponential falloff
- Edge constraints (60% deflection for boundary control points)

## Critical Issue
**The membrane_compliance = 1e3 is still too low**. Need deflections in the 2-5cm range to achieve optical control, not 4mm. The proper physics works but the material parameters need dramatic adjustment.

**Working Theory**: Fantasy physics accidentally had the right deflection scale (~10cm) for optical effects. Real physics needs material parameters tuned to achieve similar deflection magnitudes.

## Tuning Parameters to Try
- `membrane_compliance = 1e4` or `1e5` (increase by 10-100x)
- `electrode_area = 0.1` # m² (100cm² electrodes) 
- Reduce `electrode_gap = 0.030` # 30mm gap for stronger forces
- Higher test voltages in voltage patterns

## Reference: Working Fantasy Physics
Previous working parameters (for comparison):
```python
deflection_scale = 5e-11  # This gave ~10cm deflections
base_deflection = deflection_scale * (voltage**2) / (electrode_gap**2)
```

This produced unrealistic 735m calculations but ~10cm optical effects, suggesting the optical system needs deflections in that range.