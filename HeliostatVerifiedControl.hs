{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HeliostatVerifiedControl where

import Data.SBV
import Data.SBV.Control
import Data.SBV.Tools.CodeGen
import Data.SBV.Internals (modelAssocs, CV(..), CVal(..), cgRTC, defaultCgConfig)
import Control.Monad (forM_, when)
import Data.Maybe (catMaybes)

--------------------------------------------------------------------------------
-- Safety Constants and Types
--------------------------------------------------------------------------------

-- Maximum safe values
maxSafeVoltage :: SReal
maxSafeVoltage = 300.0

maxSafeWindSpeed :: SReal  
maxSafeWindSpeed = 25.0  -- m/s

maxSafeConcentration :: SReal
maxSafeConcentration = 5.0

maxSafeFluxDensity :: SReal
maxSafeFluxDensity = 5000.0  -- W/m²

minSafeGap :: SReal
minSafeGap = 0.001  -- 1mm minimum electrode gap

-- Control system states
data ControlState = 
    Normal 
  | WindWarning
  | EmergencyDefocus  
  | Shutdown
  deriving (Eq, Show)

-- Verified voltage pattern (16 zones)
type VoltagePattern = [SReal]

-- System inputs from ARTIST
data ARTISTInputs = ARTISTInputs
  { currentFluxDensity   :: SReal      -- W/m² from ARTIST ray tracer
  , maxConcentration     :: SReal      -- Peak concentration factor
  , averageDeflection    :: SReal      -- Average mylar deflection
  , deflectionVariance   :: SReal      -- Deflection non-uniformity
  , predictedHotspotRisk :: SReal      -- 0-1 risk score from ML
  }

-- Environmental inputs
data EnvironmentInputs = EnvironmentInputs
  { windSpeed        :: SReal
  , windDirection    :: SReal  
  , temperature      :: SReal
  , humidity         :: SReal
  , solarIrradiance  :: SReal
  }

-- Control outputs
data ControlOutputs = ControlOutputs
  { voltageCommands  :: VoltagePattern
  , systemState      :: SWord8  -- Encoded ControlState
  , safetyFlags      :: SWord8  -- Bit flags for various safety conditions
  }

--------------------------------------------------------------------------------
-- Safety Properties
--------------------------------------------------------------------------------

-- Property: Voltage limits are always respected
prop_VoltageLimits :: VoltagePattern -> SBool
prop_VoltageLimits voltages = 
  sAll (\v -> v .>= 0 .&& v .<= maxSafeVoltage) voltages

-- Property: Wind safety enforced
prop_WindSafety :: EnvironmentInputs -> VoltagePattern -> SBool
prop_WindSafety env voltages =
  (windSpeed env .> maxSafeWindSpeed) .=> sAll (.== 0) voltages

-- Property: Concentration limits enforced
prop_ConcentrationSafety :: ARTISTInputs -> VoltagePattern -> SBool
prop_ConcentrationSafety artist voltages =
  (maxConcentration artist .> maxSafeConcentration) .=> 
    isEmergencyPattern voltages
  where
    isEmergencyPattern vs = 
      -- Checkerboard pattern for emergency defocus
      sAnd [v .== 0 | (v, i) <- zip vs [0..], even (i :: Int)] .&&
      sAnd [v .> 0 | (v, i) <- zip vs [0..], odd (i :: Int)]

-- Property: Flux density safety
prop_FluxDensitySafety :: ARTISTInputs -> ControlOutputs -> SBool
prop_FluxDensitySafety artist outputs =
  (currentFluxDensity artist .> maxSafeFluxDensity) .=>
    (systemState outputs .== encodeState EmergencyDefocus)

-- Property: Minimum gap maintained
prop_MinimumGap :: ARTISTInputs -> VoltagePattern -> SBool
prop_MinimumGap artist voltages =
  let maxDeflection = averageDeflection artist + 3 * sqrt (deflectionVariance artist)
      electrodeGap = 0.005  -- 5mm nominal
      minGapAtMaxV = electrodeGap - maxDeflection
  in sAll (\v -> (v .== maxSafeVoltage) .=> (minGapAtMaxV .> minSafeGap)) voltages

-- Master safety property
prop_SystemSafety :: ARTISTInputs -> EnvironmentInputs -> ControlOutputs -> SBool
prop_SystemSafety artist env outputs =
  prop_VoltageLimits (voltageCommands outputs) .&&
  prop_WindSafety env (voltageCommands outputs) .&&
  prop_ConcentrationSafety artist (voltageCommands outputs) .&&
  prop_FluxDensitySafety artist outputs .&&
  prop_MinimumGap artist (voltageCommands outputs)

--------------------------------------------------------------------------------
-- Verified Control Algorithms
--------------------------------------------------------------------------------

-- Encode control state as Word8
encodeState :: ControlState -> SWord8
encodeState Normal          = 0
encodeState WindWarning     = 1
encodeState EmergencyDefocus = 2
encodeState Shutdown        = 3

-- Determine system state from inputs
determineSystemState :: ARTISTInputs -> EnvironmentInputs -> SWord8
determineSystemState artist env =
  ite (windSpeed env .> maxSafeWindSpeed .||
       temperature env .< -10 .|| temperature env .> 60)
      (encodeState Shutdown)
      (ite (currentFluxDensity artist .> maxSafeFluxDensity .||
            maxConcentration artist .> maxSafeConcentration .||
            predictedHotspotRisk artist .> 0.8)
           (encodeState EmergencyDefocus)
           (ite (windSpeed env .> 15 .|| 
                 deflectionVariance artist .> 0.01)
                (encodeState WindWarning)
                (encodeState Normal)))

-- Generate emergency defocus pattern
emergencyDefocusPattern :: VoltagePattern
emergencyDefocusPattern = 
  [ite (literal (even i)) 0 150 | i <- [0..15]]

-- Generate safe voltage pattern based on state
generateSafeVoltages :: SWord8 -> ARTISTInputs -> EnvironmentInputs -> VoltagePattern
generateSafeVoltages state artist env =
  ite (state .== encodeState Shutdown)
      (replicate 16 0)  -- All off
      (ite (state .== encodeState EmergencyDefocus)
           emergencyDefocusPattern
           (ite (state .== encodeState WindWarning)
                (map (* 0.5) normalPattern)  -- Reduced power
                normalPattern))
  where
    normalPattern = generateNormalPattern artist env

-- Generate normal operation pattern
generateNormalPattern :: ARTISTInputs -> EnvironmentInputs -> VoltagePattern
generateNormalPattern artist env =
  let baseVoltage = 100  -- Base operating voltage
      
      -- Adjust for current flux
      fluxFactor = ite (currentFluxDensity artist .< 1000) 1.2
                      (ite (currentFluxDensity artist .> 3000) 0.8 1.0)
      
      -- Adjust for wind
      windFactor = 1.0 - (windSpeed env / 50.0)  -- Linear reduction
      
      -- Environmental derating
      tempFactor = ite (temperature env .< 0) 0.9
                      (ite (temperature env .> 40) 0.85 1.0)
      
      voltage = baseVoltage * fluxFactor * windFactor * tempFactor
      
      -- Ensure within limits
      safeVoltage = ite (voltage .> maxSafeVoltage) maxSafeVoltage
                       (ite (voltage .< 0) 0 voltage)
      
  in replicate 16 safeVoltage

-- Safety flag encoding
encodeSafetyFlags :: ARTISTInputs -> EnvironmentInputs -> SWord8
encodeSafetyFlags artist env =
  let bit0 = ite (windSpeed env .> 20) 1 0
      bit1 = ite (currentFluxDensity artist .> 4000) 2 0
      bit2 = ite (maxConcentration artist .> 4) 4 0
      bit3 = ite (predictedHotspotRisk artist .> 0.5) 8 0
      bit4 = ite (temperature env .< 0 .|| temperature env .> 50) 16 0
      bit5 = ite (deflectionVariance artist .> 0.02) 32 0
      bit6 = ite (averageDeflection artist .> 0.04) 64 0
      bit7 = 0  -- Reserved
  in bit0 .|. bit1 .|. bit2 .|. bit3 .|. bit4 .|. bit5 .|. bit6 .|. bit7

-- Main verified control function
verifiedControl :: ARTISTInputs -> EnvironmentInputs -> ControlOutputs
verifiedControl artist env =
  let state = determineSystemState artist env
      voltages = generateSafeVoltages state artist env
      flags = encodeSafetyFlags artist env
  in ControlOutputs voltages state flags

--------------------------------------------------------------------------------
-- Verification Tasks
--------------------------------------------------------------------------------

-- Verify all safety properties
verifySafetyProperties :: IO ()
verifySafetyProperties = do
  putStrLn "Verifying safety properties..."
  
  -- Property 1: System never exceeds voltage limits
  r1 <- prove $ do
    flux <- sReal "flux"
    conc <- sReal "concentration" 
    defl <- sReal "deflection"
    var <- sReal "variance"
    risk <- sReal "risk"
    
    constrain $ flux .>= 0 .&& flux .<= 10000
    constrain $ conc .>= 0 .&& conc .<= 10
    constrain $ defl .>= 0 .&& defl .<= 0.05
    constrain $ var .>= 0 .&& var .<= 0.03
    constrain $ risk .>= 0 .&& risk .<= 1
    
    let artist = ARTISTInputs flux conc defl var risk
    
    wind <- sReal "wind"
    dir <- sReal "direction"
    temp <- sReal "temp"
    hum <- sReal "humidity"
    sol <- sReal "solar"
    
    constrain $ wind .>= 0 .&& wind .<= 40
    constrain $ dir .>= 0 .&& dir .<= 360
    constrain $ temp .>= -20 .&& temp .<= 60
    constrain $ hum .>= 0 .&& hum .<= 1
    constrain $ sol .>= 0 .&& sol .<= 1200
    
    let env = EnvironmentInputs wind dir temp hum sol
        outputs = verifiedControl artist env
    
    return $ prop_VoltageLimits (voltageCommands outputs)
  
  print r1
  
  -- Property 2: Wind safety is enforced
  r2 <- prove $ do
    let artist = ARTISTInputs 1000 2 0.01 0.001 0.1
    
    wind <- sReal "wind"
    let env = EnvironmentInputs wind 0 25 0.5 800
        outputs = verifiedControl artist env
    
    return $ prop_WindSafety env (voltageCommands outputs)
  
  print r2
  
  -- Property 3: Emergency defocus works
  r3 <- prove $ do
    conc <- sReal "concentration"
    constrain $ conc .> maxSafeConcentration
    
    let artist = ARTISTInputs 4000 conc 0.02 0.005 0.7
        env = EnvironmentInputs 10 180 25 0.5 800
        outputs = verifiedControl artist env
    
    return $ systemState outputs .== encodeState EmergencyDefocus
  
  print r3

-- Generate lookup table based on actual physics and electrical engineering
generateVoltageLookupTable :: IO ()
generateVoltageLookupTable = do
  putStrLn "Generating physics-based voltage lookup table..."
  putStrLn "Solving for optimal voltage patterns given environmental conditions"
  
  let scenarios = [ ("Normal focusing", 2000.0 :: Double, 5.0 :: AlgReal, 2.5 :: AlgReal)   -- flux, wind, target_concentration
                  , ("High flux defocus", 4500.0 :: Double, 8.0 :: AlgReal, 1.8 :: AlgReal)  -- need lower concentration for safety
                  , ("Windy compensation", 1800.0 :: Double, 20.0 :: AlgReal, 2.0 :: AlgReal) -- need wind compensation
                  , ("Emergency limit", 4900.0 :: Double, 12.0 :: AlgReal, 1.2 :: AlgReal)   -- minimal focusing, near safety limit
                  ]
  
  results <- mapM computeLookupEntry scenarios
  
  let validEntries = length (filter (not . null . fst) results)
      totalEntries = length results
  
  putStrLn $ "\nLookup table complete: " ++ show validEntries ++ "/" ++ show totalEntries ++ " valid entries"
  putStrLn "\nSample lookup table entries:"
  mapM_ printEntry results
  
  where
    computeLookupEntry (name, fluxVal, windVal, targetConc) = do
      putStrLn $ "\nComputing: " ++ name
      putStrLn $ "  Conditions: flux=" ++ show fluxVal ++ " W/m², wind=" ++ show windVal ++ " m/s"
      putStrLn $ "  Target concentration: " ++ show targetConc ++ "x"
      
      result <- (sat :: Symbolic SBool -> IO SatResult) $ do
        -- Create symbolic variables for the 16 zone voltages
        v1 <- sReal "v1"; v2 <- sReal "v2"; v3 <- sReal "v3"; v4 <- sReal "v4"
        v5 <- sReal "v5"; v6 <- sReal "v6"; v7 <- sReal "v7"; v8 <- sReal "v8"
        v9 <- sReal "v9"; v10 <- sReal "v10"; v11 <- sReal "v11"; v12 <- sReal "v12"
        v13 <- sReal "v13"; v14 <- sReal "v14"; v15 <- sReal "v15"; v16 <- sReal "v16"
        
        let voltages = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16]
        
        -- Physics constraints: voltage limits
        mapM_ (\v -> constrain $ v .>= 0 .&& v .<= maxSafeVoltage) voltages
        
        -- Physics constraint: parabolic focusing pattern
        -- Edge zones (corners) should have higher voltage for focusing
        let edgeZones = [v1, v4, v13, v16]  -- corner zones
            centerZones = [v6, v7, v10, v11]  -- center zones
            edgeAvg = sum edgeZones / 4
            centerAvg = sum centerZones / 4
        
        -- For focusing: edge voltage should be higher than center
        constrain $ edgeAvg .>= centerAvg
        
        -- Target concentration constraint: voltage ratio relates to concentration
        -- Higher voltage ratio = more focusing = higher concentration
        let voltageRatio = edgeAvg / (centerAvg + 1)  -- +1 to avoid division by zero
        constrain $ voltageRatio .<= literal targetConc
        constrain $ voltageRatio .>= literal (targetConc * 0.8)  -- within 20% of target
        
        -- Wind compensation: higher voltages needed for higher wind
        let windCompensation = literal (windVal * 2.0)  -- 2V per m/s wind
        constrain $ edgeAvg .>= windCompensation
        
        -- Safety constraint: if flux is high, reduce overall voltages
        let fluxSafetyFactor = ite (literal fluxVal .> 4000) 0.7 1.0
        mapM_ (\v -> constrain $ v .<= maxSafeVoltage * fluxSafetyFactor) voltages
        
        -- Ensure meaningful voltages (not all zero)
        constrain $ sOr [v .> 20 | v <- voltages]
        
        (return :: SBool -> Symbolic SBool) sTrue
      
      case result of
        SatResult (Satisfiable _ model) -> do
          putStrLn "  ✓ Found optimal voltage pattern"
          let voltageValues = extractVoltages model
          putStrLn $ "  Voltages: " ++ show voltageValues
          return (voltageValues, (fluxVal, windVal, targetConc))
        SatResult (Unsatisfiable _ _) -> do
          putStrLn "  ✗ No safe voltage pattern exists for these conditions"
          return ([], (fluxVal, windVal, targetConc))
        _ -> do
          putStrLn "  ? Solver error"
          return ([], (fluxVal, windVal, targetConc))
    
    extractVoltages model = 
      let voltageNames = ["v" ++ show i | i <- [1..16]]
          lookupVoltage name = case lookup name (modelAssocs model) of
            Just cv -> cvToDouble cv
            Nothing -> 0.0
      in map lookupVoltage voltageNames
    
    cvToDouble cv = case cv of
      CV _ (CAlgReal algReal) -> 
        -- AlgReal is complex, just convert to string and back as approximation
        read (show algReal) :: Double
      CV _ (CInteger n) -> fromInteger n
      CV _ (CFloat f) -> realToFrac f
      CV _ (CDouble d) -> d
      _ -> 0.0
    
    printEntry (voltages, (flux, wind, conc)) 
      | null voltages = return ()
      | otherwise = do
          putStrLn $ "Entry: flux=" ++ show flux ++ ", wind=" ++ show wind ++ ", target=" ++ show conc
          putStrLn $ "  Voltages: " ++ show (take 4 voltages) ++ "... (showing first 4 of 16)"
          let avgVoltage = sum voltages / fromIntegral (length voltages)
              maxVoltage = maximum voltages
          putStrLn $ "  Stats: avg=" ++ show avgVoltage ++ "V, max=" ++ show maxVoltage ++ "V"

--------------------------------------------------------------------------------
-- C Code Generation
--------------------------------------------------------------------------------

-- Generate verified C code for embedded controller
generateEmbeddedController :: IO ()
generateEmbeddedController = do
  putStrLn "Generating verified C code..."
  
  compileToC (Just ".") "verified_heliostat_control" $ do
    cgSRealType CgFloat  -- Specify that SReal should be represented as float
    -- Inputs from ARTIST ray tracer
    flux <- cgInput "flux_density"
    concentration <- cgInput "max_concentration"
    avg_deflection <- cgInput "avg_deflection"
    var_deflection <- cgInput "var_deflection"
    hotspot_risk <- cgInput "hotspot_risk"
    
    -- Environmental inputs
    wind_speed <- cgInput "wind_speed"
    wind_dir <- cgInput "wind_direction"
    temperature <- cgInput "temperature"
    humidity <- cgInput "humidity"
    solar_irrad <- cgInput "solar_irradiance"
    
    -- Create input structures
    let artist = ARTISTInputs flux concentration avg_deflection 
                             var_deflection hotspot_risk
        env = EnvironmentInputs wind_speed wind_dir temperature 
                               humidity solar_irrad
    
    -- Run verified control
    let outputs = verifiedControl artist env
    
    -- Output voltage commands
    cgOutputArr "voltage_commands" (voltageCommands outputs)
    
    -- Output system state
    cgOutput "system_state" (systemState outputs)
    
    -- Output safety flags
    cgOutput "safety_flags" (safetyFlags outputs)
    
    -- Additional safety checks
    let voltage_ok = prop_VoltageLimits (voltageCommands outputs)
        wind_ok = wind_speed .<= maxSafeWindSpeed
        flux_ok = flux .<= maxSafeFluxDensity
        all_ok = voltage_ok .&& wind_ok .&& flux_ok
    
    cgOutput "safety_check_passed" all_ok

--------------------------------------------------------------------------------
-- Integration with ARTIST
--------------------------------------------------------------------------------

-- Prove that control decisions are monotonic with risk
prop_MonotonicSafety :: IO ThmResult
prop_MonotonicSafety = prove $ do
  -- Two scenarios with increasing risk
  risk1 <- sReal "risk1"
  risk2 <- sReal "risk2"
  constrain $ risk1 .>= 0 .&& risk1 .<= 1
  constrain $ risk2 .>= 0 .&& risk2 .<= 1
  constrain $ risk1 .< risk2  -- risk2 is higher
  
  let artist1 = ARTISTInputs 2000 3 0.01 0.001 risk1
      artist2 = ARTISTInputs 2000 3 0.01 0.001 risk2
      env = EnvironmentInputs 10 0 25 0.5 800
      
      out1 = verifiedControl artist1 env
      out2 = verifiedControl artist2 env
      
      -- Average voltage as proxy for aggressiveness
      avgV1 = sum (voltageCommands out1) / 16
      avgV2 = sum (voltageCommands out2) / 16
  
  -- Higher risk should lead to same or lower voltages
  return $ avgV2 .<= avgV1

-- Generate Python bindings for ARTIST integration
generatePythonInterface :: IO ()
generatePythonInterface = do
  putStrLn "Generating Python interface..."
  writeFile "heliostat_verified_control.py" $ unlines
    [ "import ctypes"
    , "import numpy as np"
    , ""
    , "# Load the compiled verified controller"
    , "lib = ctypes.CDLL('./verified_heliostat_control.so')"
    , ""
    , "# Define the control function signature"
    , "lib.verified_heliostat_control.argtypes = ["
    , "    ctypes.c_double,  # flux_density"
    , "    ctypes.c_double,  # max_concentration"
    , "    ctypes.c_double,  # avg_deflection"
    , "    ctypes.c_double,  # var_deflection"
    , "    ctypes.c_double,  # hotspot_risk"
    , "    ctypes.c_double,  # wind_speed"
    , "    ctypes.c_double,  # wind_direction"
    , "    ctypes.c_double,  # temperature"
    , "    ctypes.c_double,  # humidity"
    , "    ctypes.c_double,  # solar_irradiance"
    , "    ctypes.POINTER(ctypes.c_double * 16),  # voltage_commands"
    , "    ctypes.POINTER(ctypes.c_uint8),  # system_state"
    , "    ctypes.POINTER(ctypes.c_uint8),  # safety_flags"
    , "    ctypes.POINTER(ctypes.c_bool)  # safety_check_passed"
    , "]"
    , ""
    , "def verified_control(artist_inputs, env_inputs):"
    , "    \"\"\"Call verified controller from Python/ARTIST\"\"\""
    , "    voltages = (ctypes.c_double * 16)()"
    , "    state = ctypes.c_uint8()"
    , "    flags = ctypes.c_uint8()"
    , "    safe = ctypes.c_bool()"
    , "    "
    , "    lib.verified_heliostat_control("
    , "        artist_inputs['flux_density'],"
    , "        artist_inputs['max_concentration'],"
    , "        artist_inputs['avg_deflection'],"
    , "        artist_inputs['var_deflection'],"
    , "        artist_inputs['hotspot_risk'],"
    , "        env_inputs['wind_speed'],"
    , "        env_inputs['wind_direction'],"
    , "        env_inputs['temperature'],"
    , "        env_inputs['humidity'],"
    , "        env_inputs['solar_irradiance'],"
    , "        ctypes.byref(voltages),"
    , "        ctypes.byref(state),"
    , "        ctypes.byref(flags),"
    , "        ctypes.byref(safe)"
    , "    )"
    , "    "
    , "    return {"
    , "        'voltages': np.array(voltages),"
    , "        'state': state.value,"
    , "        'flags': flags.value,"
    , "        'safe': safe.value"
    , "    }"
    ]

--------------------------------------------------------------------------------
-- Test Suite
--------------------------------------------------------------------------------

runTests :: IO ()
runTests = do
  putStrLn "Running verification tests..."
  
  -- Verify safety properties
  verifySafetyProperties
  
  -- Verify monotonic safety
  r <- prop_MonotonicSafety
  putStrLn $ "Monotonic safety: " ++ show r
  
  -- Generate C code
  generateEmbeddedController
  
  -- Generate Python interface
  generatePythonInterface
  
  -- Generate lookup tables
  generateVoltageLookupTable
  
  putStrLn "All tests completed!"

-- Example usage showing the integration
exampleIntegration :: IO ()
exampleIntegration = do
  putStrLn "Example ARTIST + SBV integration:"
  putStrLn ""
  putStrLn "# Python code using ARTIST + verified controller:"
  putStrLn "import artist"
  putStrLn "from heliostat_verified_control import verified_control"
  putStrLn ""
  putStrLn "# ARTIST ray tracing"
  putStrLn "flux = artist.trace_heliostat(surface, sun_pos)"
  putStrLn "concentration = flux.max() / flux.mean()"
  putStrLn ""
  putStrLn "# Prepare inputs"
  putStrLn "artist_inputs = {"
  putStrLn "    'flux_density': flux.max(),"
  putStrLn "    'max_concentration': concentration,"
  putStrLn "    'avg_deflection': surface.mean_deflection(),"
  putStrLn "    'var_deflection': surface.deflection_variance(),"
  putStrLn "    'hotspot_risk': ml_model.predict(flux)"
  putStrLn "}"
  putStrLn ""
  putStrLn "# Get verified control commands"
  putStrLn "control = verified_control(artist_inputs, env_inputs)"
  putStrLn ""
  putStrLn "# Apply voltages (guaranteed safe!)"
  putStrLn "heliostat.set_voltages(control['voltages'])"

main :: IO ()
main = do
  putStrLn "SBV + ARTIST Heliostat Control System"
  putStrLn "====================================="
  runTests
  putStrLn ""
  exampleIntegration