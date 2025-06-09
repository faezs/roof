import ctypes
import numpy as np

# Load the compiled verified controller
lib = ctypes.CDLL('./verified_heliostat_control.so')

# Define the control function signature
lib.verified_heliostat_control.argtypes = [
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

def verified_control(artist_inputs, env_inputs):
    """Call verified controller from Python/ARTIST"""
    voltages = (ctypes.c_double * 16)()
    state = ctypes.c_uint8()
    flags = ctypes.c_uint8()
    safe = ctypes.c_bool()
    
    lib.verified_heliostat_control(
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
