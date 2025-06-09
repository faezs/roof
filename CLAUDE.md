# Heliostat Project Development Guide

## Build/Test Commands
- Build: `nix build` - Uses flake.nix to build the project
- Run type checker: `vehicle check heliostat.vcl`
- Validate: `vehicle verify heliostat.vcl`
- Run simulation: `vehicle run heliostat.vcl`
- Generate object file: `vehicle compile heliostat.vcl -o heliostat.vclo`
- View inference rules: `vehicle show-rules heliostat.vcl`

## Code Style Guidelines
- Use tabs for indentation in Vehicle Language (.vcl) files
- Prefer descriptive variable names with camelCase (e.g., `heliostatWidth`)
- Group related physical constants and parameters together
- Document properties with detailed comments
- Use Vector types for fixed-size collections
- Wrap physical properties in appropriate types (Temperature, HeatFlux, etc.)
- Follow mathematical notation conventions for physics variables
- Use "smooth" functions for derivatives across discontinuities
- Prefer pure functional style when possible
- Define and use appropriate domain-specific types
- Document parameter units in comments (e.g., W/m², K, Pa)