# Heliostat Verified Control System Makefile

.PHONY: all verify build test clean

# Build targets
all: verify build test

# Generate and verify the control system
verify:
	@echo "Generating verified control system..."
	nix develop --command ghc --make HeliostatVerifiedControl.hs -o verification
	./verification

# Build the C library from generated code
build:
	@echo "Building C library..."
	@if [ -f verified_heliostat_control.c ]; then \
		gcc -shared -fPIC -O3 -o verified_heliostat_control.so verified_heliostat_control.c -lm; \
		echo "Built verified_heliostat_control.so"; \
	else \
		echo "Warning: No C code generated yet. Run 'make verify' first."; \
	fi

# Test the Python integration
test:
	@echo "Testing Python integration..."
	nix develop --command python3 artist_integration.py

# Clean build artifacts
clean:
	rm -f verification
	rm -f verified_heliostat_control.c
	rm -f verified_heliostat_control.h
	rm -f verified_heliostat_control.so
	rm -f heliostat_verified_control.py
	rm -f *.hi *.o
	rm -f heliostat_data.npz

# Quick test without full verification
quick-test:
	@echo "Running quick Python test..."
	nix develop --command python3 -c "import sys; sys.path.append('.'); from artist_integration import *; import logging; logging.basicConfig(level=logging.INFO); config = SystemConfig(); controller = HeliostatController(config); print('✓ Successfully created heliostat controller'); print('✓ System ready for operation')"

# Build Vehicle Language heliostat model
vehicle:
	@echo "Building Vehicle heliostat model..."
	vehicle check heliostat.vcl
	vehicle compile heliostat.vcl -o heliostat.vclo

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build and verify everything"
	@echo "  verify     - Generate and verify control system"
	@echo "  build      - Build C library from generated code"
	@echo "  test       - Test Python integration"
	@echo "  quick-test - Quick test without full verification"
	@echo "  vehicle    - Build Vehicle language model"
	@echo "  clean      - Clean build artifacts"
	@echo "  help       - Show this help"