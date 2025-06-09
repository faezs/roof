-- Heliostat Physics Module: Rigorous Physical Modeling


-- Core Types
type R3 = Tensor Rat [3]
x = 0
y = 1
z = 2

type R4 = Tensor Rat [4]
t = 3

@parameter
im_w : Nat
@parameter
im_h : Nat

type Image = Tensor Rat [im_w, im_h]
type Watts = Rat
type WindSpeed = R3
type SolarFlux = Image 
type Wind = R4



-- The monoid
sums : forallT {n} . Tensor Rat n -> Rat
sums xs = fold (\a b -> a + b) 0 xs

sum : forallT {f : Type -> Type} . {{HasFold f}} -> f Rat -> Rat
sum xa ya = xa + ya



-- sumImage : Image -> Rat
-- sumImage xs = sum (map sum xs)

-- Math?
-- Natural logarithm via Taylor series
-- ln(1 + x) = x - x²/2 + x³/3 - x⁴/4 + ...
-- Valid for -1 < x < 1

-- cnvrg : forallT {a : Type} . {{HasSub a}} {{HasAdd a}} {{HasDiv a}} -> a -> a
cnvrg xs = ((xs - 1) / (xs + 1))

sqr xs = xs * xs

ln : Rat -> Rat
ln xs = 2 * ((cnvrg xs) + (sqr (cnvrg xs)) * (sqr (cnvrg xs)) /3 + (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (cnvrg xs) / 5 + (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (cnvrg xs) /7 + (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (sqr (cnvrg xs)) * (cnvrg xs) / 9)

-- Exponential function via Taylor series
-- e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
exp : Rat -> Rat
exp xs =
  1 + xs + xs*xs/2 + xs*xs*xs/6 + xs*xs*xs*xs/24 + xs*xs*xs*xs*xs/120

-- General exponentiation for positive base
pow : Rat -> Rat -> Rat
pow base exponent = exp (exponent * ln base)

square : Rat -> Rat
square ax = ax * ax 

-- Square root via Newton-Raphson iteration
sqrt : Rat -> Rat
sqrt xs = ((((xs / 2 + xs / xs / 2) / 2 + xs / (xs / 2 + xs / xs / 2) / 2) / 2) + xs / (((xs / 2 + xs / xs / 2) / 2 + xs / (xs / 2 + xs / xs / 2) / 2) / 2)) / 2
--  let guess = xs / 2 in
--  let improved = (xs / 2 + xs / xs / 2) / 2 in
--  let better = (((xs / 2 + xs / xs / 2) / 2 + xs / (xs / 2 + xs / xs / 2) / 2) / 2) in




-- Smooth Logic?


-- Smooth approximation to max function
smoothMax : Rat -> Rat -> Rat -> Rat
smoothMax a b sharpness = (a * exp (sharpness * a) + b * exp (sharpness * b)) / (exp (sharpness * a) + exp (sharpness * b))
  -- let expA = exp (sharpness * a) in
  -- let expB = exp (sharpness * b) in
  

-- Smooth approximation to step function (0 to 1 transition)
smoothStep : Rat -> Rat -> Rat
smoothStep xa sharpness =
  1 / (1 + exp (-sharpness * xa))

smoothFloor : Rat -> Rat
smoothFloor a = a
  -- Sum of smooth step functions
  -- let maxInt = 100 in  -- Reasonable range for hours/days
  -- (sum (foreach i . smoothStep (a 100))


-- Smooth modulo function for periodic values
smoothMod : Rat -> Rat -> Rat
smoothMod a b = a - b * (smoothFloor a/b)
  -- Use smooth floor: a mod b = a - b * floor(a/b)
  

-- Smooth absolute value
smoothAbs : Rat -> Rat
smoothAbs a = sqrt (a * a + 0.0001)

-- Smooth sign function (-1 to 1)
smoothSign : Rat -> Rat
smoothSign a = a / sqrt (a * a + 0.0001)

zero : forallT {@0 n} . Vector Rat n
zero = foreach i . 0
  

-- trig and linear alg
e = 2.7182818

pi = 3.141592


sin xs = xs - xs*xs*xs/6
cos xs = 1 - xs*xs/2
tan xs = sin xs / cos xs 
asin xs = xs + 1/2 * xs*xs*xs + (3/8 * (xs*xs*xs*xs*xs)/5) 
acos xs = pi/2 - asin xs

abs xs = if xs < 0 then -xs else xs

-- addTensors : forallT {@0 dims} . {{HasAdd A B C}} -> Tensor A dims -> Tensor B dims -> Tensor C dim
-- addTensors xs ys = zipWith (\x y -> x + y)

-- addVector : forallT {@0 n} . {{HasAdd A B C}} -> Vector A n -> Vector B n -> Vector C n
-- addVector = zipWith (\x y -> x + y)


dotProduct a b = a ! x * b ! x + a ! y * b ! y + a ! z * b ! z
crossProduct a b = 
      [ a ! y * b ! z - a ! z * b ! y
      , a ! z * b ! x - a ! x * b ! z  
      , a ! x * b ! y - a ! y * b ! x ]



atan2 ya xa =
  -- Use smooth approxaimation that handles all quadrants
  -- let r = sqrt (xa * xa + ya * ya + 0.0001) in  -- Add small epsilon for stability
  -- let theta = asin (ya / r) in
  -- Adjust for quadrant using smooth sign of xa
  asin (ya / sqrt (xa * xa + ya * ya + 0.0001)) + (1 - smoothSign xa) * smoothSign ya * pi / 2


tolerance = 0.01  -- 1% error tolerance

norm v = sqrt (dotProduct v v)
normalize v = foreach i . v ! i / norm v


distance p1 p2 = norm (p1 - p2)
extractNormal rot = [rot ! 2 ! 0, rot ! 2 ! 1, rot ! 2 ! 2]


-- Physical Constants

c : Rat
c = 299792458  -- Speed of light (m/s)

h : Rat
h = 6.626 * (pow e (-34))  -- Planck's constant (J⋅s) 

k_B : Rat
k_B = 1.381 * pow e (-23)  -- Boltzmann constant (J/K)

sigma : Rat
sigma = 5.67 * pow e (-8)  -- Stefan-Boltzmann constant (W⋅m⁻²⋅K⁻⁴)

-- Solar Constants
solar_constant : Rat
solar_constant = 1361  -- W/m² at 1 AU

sun_temperature : Rat
sun_temperature = 5778  -- K (effective temperature)

-- Material Properties
type MaterialProperties = Tensor Rat [6]
reflectivity_idx = 0
absorptivity_idx = 1
emissivity_idx = 2
thermal_conductivity_idx = 3
specific_heat_idx = 4
density_idx = 5

@dataset
mirror_material : MaterialProperties

@dataset
receiver_material : MaterialProperties

-- Geometric Configuration
type Point = Tensor Rat [4]
lat = 0
long = 1
height = 2
pt = 3


type LocalCoordinateFrame = Point -> R4
type SpaceEmbedding = R4 -> R3
type TimeEmbedding = R4 -> Rat

@parameter
o : Nat
@parameter
n : Nat
 
type NurbsSurface = Tensor Rat [o, n]

-- Input tensor with full physical state
type InputTensor = Tensor Rat [ 4      -- spacetime
                              , 3      -- windflow
                              , 28, 28 -- Light Flux (spectral irradiance)
                              , 10     -- NURBS surface geometry
                              ]

@dataset
sensors : InputTensor

-- Output: Rotation matrix for heliostat orientation
type Rotation = Tensor Rat [3, 3]

@network
heliostatController : InputTensor -> Rotation



-- Electromagnetic Wave Propagation
type SpectralIrradiance = Rat -> Rat  -- W⋅m⁻²⋅nm⁻¹ as function of wavelength


planckRadiation : Rat -> Rat -> Rat
planckRadiation temperature wavel =
  -- let numerator = 2 * h * c * c in
  -- let denominator = (pow wavel 5) * (exp (h * c / (wavel * k_B * temperature)) - 1) in
  (2 * h * c * c) / ((pow wavel 5) * (exp (h * c / (wavel * k_B * temperature)) - 1))


projectPointToGround : R3 -> R3 -> R3
projectPointToGround point dir =
  -- Ray equation: p + t*d intersects ground when z component = 0
  -- Smooth clamping to avoid division issues
  -- let safeZ = smoothMax (-0.1) (smoothMax dir ! z (-dir ! z)) 5 in
  -- let ta = (-(point ! z) / smoothMax (-0.1) (smoothMax dir ! z (-dir ! z))) 5 in
  [((point ! x) + (-(point ! z) / (max (-0.1) (max (dir ! z) (-dir ! z)))) * dir ! x),
   ((point ! y) + (-(point ! z) / (max (-0.1) (max (dir ! z) (-dir ! z)))) * dir ! y),
   0]

-- Project shadow of a rectangular heliostat onto ground
projectShadow : R3 -> R3 -> Vector R3 4
projectShadow pos dir =
    let
      heliostatWidth = 2,
      heliostatHeight = 2,
      -- Corner offsets from center
      corner1 = [-heliostatWidth/2, -heliostatHeight/2, 0],
      corner2 = [heliostatWidth/2, -heliostatHeight/2, 0],
      corner3 = [heliostatWidth/2, heliostatHeight/2, 0],
      corner4 = [-heliostatWidth/2, heliostatHeight/2, 0],
      cornerOffset = sum [corner1, corner2, corner3, corner4]
    in foreach i . projectPointToGround ([pos ! x + cornerOffset ! x, 
                         pos ! y + cornerOffset ! y, 
                         pos ! z + cornerOffset ! z]) dir

distanceToEdge : R3 -> R3 -> R3 -> Rat
distanceToEdge p v1 v2 =
  let edge = [v2 ! x - v1 ! x, v2 ! y - v1 ! y, 0] in
  let edgeLength = sqrt (square (edge ! x) + square (edge ! y)) in
  let toPoint = [p ! x - v1 ! x, p ! y - v1 ! y, 0] in
  -- Project point onto edge
  let ta = (toPoint ! x * edge ! x + toPoint ! y * edge ! y) / (edgeLength * edgeLength) in
  -- Smooth clamping between 0 and 1
  let tClamped = smoothStep ta 10 * smoothStep (1 - ta) 10 in
  let projX = v1 ! x + tClamped * edge ! x in
  let projY = v1 ! y + tClamped * edge ! y in
  sqrt (square (p ! x - projX) + square (p ! y - projY))

-- Minimum distance from point to polygon (smooth approximation)
minDistanceToPolygon : R3 -> Vector R3 4 -> Rat
minDistanceToPolygon point polygon =
  let d0 = distanceToEdge point (polygon ! 0) (polygon ! 1) in
  let d1 = distanceToEdge point (polygon ! 1) (polygon ! 2) in
  let d2 = distanceToEdge point (polygon ! 2) (polygon ! 3) in
  let d3 = distanceToEdge point (polygon ! 3) (polygon ! 0) in
  -- Smooth minimum using negative smooth max
  -smoothMax (-d0) (smoothMax (-d1) (smoothMax (-d2) (-d3) 10) 10) 10

-- Compute overlap area between shadow polygon and heliostat rectangle
computeOverlap : Vector R3 4 -> R3 -> Rat
computeOverlap s ha =
  let heliostatWidth = 2 in
  let heliostatHeight = 2 in
  -- Use smooth distance functions for overlap
  let centerDist = minDistanceToPolygon ha s in
  let normalizedDist = centerDist / sqrt (square heliostatWidth + square heliostatHeight) in
  -- Smooth transition from full overlap (4) to no overlap (0)
  4 * smoothStep (0.5 - normalizedDist) 10

-- Compute zenith angle from point coordinates and time
computeZenithAngle : Point -> Rat
computeZenithAngle p =
  let latitude = p ! lat * pi / 180 in
  let longitude = p ! long * pi / 180 in
  let hoursSinceMidnight = p ! t - smoothFloor (p ! t / 24) * 24 in
  let solarTime = hoursSinceMidnight + longitude * 12 / pi in
  let hourAngle = 15 * (solarTime - 12) * pi / 180 in
  let dayOfYear = smoothFloor (p ! t / 24) in
  let declination = 23.45 * sin (360 * (284 + dayOfYear) / 365 * pi / 180) * pi / 180 in
  let cosZenith = sin latitude * sin declination + 
                  cos latitude * cos declination * cos hourAngle in
  -- Smooth clamping between -1 and 1 for acos domain
  let cosZenithClamped = smoothMax (-0.999) (smoothMax cosZenith 0.999 10) 10 in
  acos cosZenithClamped * 180 / pi


-- Compute air mass using smooth Kasten-Young formula
computeAirMass : Point -> Rat
computeAirMass p =
  let zenithAngle = computeZenithAngle p in
  let zenithRad = zenithAngle * pi / 180 in
  -- Smooth transition at horizon (90 degrees)
  let belowHorizon = smoothStep (90 - zenithAngle) 0.1 in
  let kastenYoung = 1 / cos zenithRad + 0.50572 * pow (96.07995 - zenithAngle) (-1.6364) in
  let horizonLimit = 40 in
  -- Smooth interpolation between formula and horizon limit
  belowHorizon * kastenYoung + (1 - belowHorizon) * horizonLimit


-- Atmospheric Transmission
atmosphericTransmission : R3 -> Rat -> Rat
atmosphericTransmission position wavelength =
  let altitude = position ! z in
  let airMass = computeAirMass position in
  let rayleigh = exp (-0.008735 * (pow (wavelength / 1000) (-4.08)) * airMass) in
  let aerosol = exp (-0.1 * (pow (wavelength / 1000) (-1.3)) * airMass) in
  let water = exp (-0.0075 * airMass) in  -- Simplified water vapor
  rayleigh * aerosol * water


-- Ray Tracing for Mirror Reflection
type Ray = Tensor Rat [6]  -- origin (3) + direction (3)

reflectRay : Ray -> R3 -> MaterialProperties -> Ray
reflectRay incident normal material =
  let dir = [incident ! 3, incident ! 4, incident ! 5] in
  let na = normalize normal in
  let cosTheta = -(dotProduct dir na) in
  let reflected = foreach i . dir ! i + 2 * cosTheta * na ! i in
  let intensity = incident ! 6 * material ! reflectivity_idx in
  [incident ! 0, incident ! 1, incident ! 2, reflected ! x, reflected ! y, reflected ! z]

-- Heat Transfer Equations
type Temperature = Rat
type HeatFlux = Rat

-- Radiation heat transfer
radiationHeatTransfer : Temperature -> Temperature -> Rat -> HeatFlux
radiationHeatTransfer t1 t2 area =
  sigma * area * (pow t1 4 - pow t2 4)

convectionCoefficient : WindSpeed -> Rat
convectionCoefficient wind =
  let v = norm wind in
  2.8 + 3.0 * sqrt v  -- Empirical correlation

-- Convection heat transfer  
convectionHeatTransfer : Temperature -> Temperature -> Rat -> WindSpeed -> HeatFlux
convectionHeatTransfer surface ambient area wind =
  let ha = convectionCoefficient wind in
  ha * area * (surface - ambient)

-- Conduction heat transfer
conductionHeatTransfer : Temperature -> Temperature -> Rat -> Rat -> MaterialProperties -> HeatFlux
conductionHeatTransfer t1 t2 area thickness material =
  material ! thermal_conductivity_idx * area * (t1 - t2) / thickness

-- Energy Balance at Receiver
type ReceiverState = Tensor Rat [4]
receiver_temp_idx = 0
absorbed_power_idx = 1
emitted_power_idx = 2
conducted_power_idx = 3

-- @parameter(default=10)
receiverArea : Rat -- m^2
receiverArea = 10

solveForTemperature : Watts -> WindSpeed -> Temperature -> MaterialProperties -> Temperature
solveForTemperature q w ta mat = ta + q / (receiverArea * convectionCoefficient w)

computeReceiverEnergyBalance : HeatFlux -> WindSpeed -> Temperature -> MaterialProperties -> ReceiverState
computeReceiverEnergyBalance incidentFlux wind ambient material =
      -- Newton-Raphson iteration for energy balance -- Simplified 
  let absorbed = incidentFlux * material ! absorptivity_idx in
  let temp = solveForTemperature absorbed wind ambient material in
  let emitted = radiationHeatTransfer temp ambient receiverArea in
  let convected = convectionHeatTransfer temp ambient receiverArea wind in
  [temp, absorbed, emitted, convected]
    

-- Heliostat Field Optimization
type HeliostatField = Vector Point 100  -- Position of each heliostat

@parameter
field_layout : HeliostatField

-- @parameter
heliostatArea : Rat -- m^2
heliostatArea = 4



-- Compute shading and blocking factors
shadingFactor : Point -> Point -> R3 -> Rat
shadingFactor heliostat1 heliostat2 sunDirection =
  let shadow = projectShadow heliostat1 sunDirection in
  let overlap = computeOverlap shadow heliostat2 in
  1 - overlap / heliostatArea
    

-- Cosine efficiency
cosineEfficiency : R3 -> R3 -> Rat
cosineEfficiency sunDirection mirrorNormal =
  abs (dotProduct sunDirection mirrorNormal)

-- Atmospheric attenuation over distance
atmosphericAttenuation : Rat -> Rat
atmosphericAttenuation dist = exp (-0.0001 * dist)  -- Simplified model

-- @parameter
spillageEfficiency : Rat  -- 95% of reflected light hits receiver
spillageEfficiency = 0.95

computeHeliostatNormal : Point -> R3 -> R3 -> R3
computeHeliostatNormal heliostatPos sunDir receiverPos =
  let toSun = normalize sunDir in
  let toReceiver = normalize (receiverPos - heliostatPos) in
  normalize (toSun + toReceiver)  -- Halfway vector


-- Total optical efficiency
opticalEfficiency : Point -> R3 -> R3 -> MaterialProperties -> Rat
opticalEfficiency heliostatPos sunDir receiverPos material =
  let cosEff = cosineEfficiency sunDir (computeHeliostatNormal heliostatPos sunDir receiverPos) in
  let dist = distance heliostatPos receiverPos in
  let atmAtten = atmosphericAttenuation dist in
  let reflectivity = material ! reflectivity_idx in
  cosEff * atmAtten * reflectivity * spillageEfficiency   



-- Wind Load Analysis
type WindLoad = Tensor Rat [3]  -- Force vector

-- @parameter
airDensity : Rat   -- kg/m³ at sea level
airDensity = 1.225


angleOfAttack : WindSpeed -> R3 -> Rat
angleOfAttack wind normal =
  acos (dotProduct wind normal / (norm wind * norm normal))

dragCoefficient : Rat -> Rat
dragCoefficient angle =
  1.28 * sin angle + 0.3  -- Empirical for flat plate


computeWindLoad : WindSpeed -> Rotation -> Rat -> WindLoad
computeWindLoad wind rotation area =
  let normal = extractNormal rotation in
  let v = norm wind in
  let angle = angleOfAttack wind normal in
  let Cd = dragCoefficient angle in
  let pressure = 0.5 * airDensity * v * v in
  let force = pressure * Cd * area in
  foreach i . force * normal ! i




-- Structural Analysis
type Stress = Rat
type Moment = R3

computeStructuralLoads : WindLoad -> Point -> Stress
computeStructuralLoads load mountPoint =
  let mountHeight = [0, 0, 2] in -- 2m high mount
  let sectionModulus = 0.001 in  -- m³ for support structure
  let moment = crossProduct (mountHeight) load in
  let stress = norm moment / sectionModulus in
  stress
    

-- Properties: Physical Constraints

-- Conservation of energy

angleDeviation : R3 -> R3 -> Rat
angleDeviation v1 v2 = acos (dotProduct v1 v2)

-- Utility functions
extractSpacetime : InputTensor -> R4
extractSpacetime inp = foreach i . inp ! i ! 0 ! 0 ! 0

extractWindflow : InputTensor -> R3
extractWindflow inp = foreach i . inp ! 4 ! i ! 0 ! 0

extractSolarFlux : InputTensor -> Image
extractSolarFlux inp = foreach i . foreach j . inp ! 4 ! 3 ! i ! j


-- @parameter
receiver_position : Tensor Rat [3]  -- Example position
receiver_position = [0, 0, 10]

-- Convert time to Julian Day Number (smooth)
timeToJulianDay : Point -> Rat
timeToJulianDay p =
  -- Assuming t is hours since epoch (Jan 1, 2000, 12:00 UTC)
  let epochJD = 2451545.0 in  -- J2000.0 epoch
  let daysSinceEpoch = p ! t / 24 in
  epochJD + daysSinceEpoch


-- Calculate sun's declination and hour angle
calculateSunAngles : Rat -> Point -> Tensor Rat [2]
calculateSunAngles jd p =
  -- Number of days since J2000.0
  let na = jd - 2451545.0 in
  
  -- Mean longitude of the Sun (degrees)
  let L = smoothMod (280.460 + 0.9856474 * na) 360 in
  
  -- Mean anomaly of the Sun (radians)
  let g = smoothMod (357.528 + 0.9856003 * na) 360 * pi / 180 in
  
  -- Ecliptic longitude of the Sun (radians)
  let lambda = (L + 1.915 * sin g + 0.020 * sin (2 * g)) * pi / 180 in
  
  -- Obliquity of the ecliptic (radians)
  let epsilon = (23.439 - 0.0000004 * na) * pi / 180 in
  
  -- Sun's declination (radians)
  let declination = asin (sin epsilon * sin lambda) in
  
  -- Equation of time (minutes)
  let eqTime = 4 * (L - 0.0057183 - atan2 (tan lambda) (cos epsilon)) in
  
  -- Hour angle (radians)
  let localTime = smoothMod (p ! t) 24 in
  let solarTime = localTime + eqTime / 60 + p ! long / 15 in
  let hourAngle = (solarTime - 12) * 15 * pi / 180 in
  
  [declination, hourAngle]


-- Convert sun angles to topocentric Cartesian coordinates
sunToTopocentric : Rat -> Rat -> Rat -> R3
sunToTopocentric latitude declination hourAngle =
  -- Elevation angle (altitude)
  let elevation = asin (sin latitude * sin declination + 
                       cos latitude * cos declination * cos hourAngle) in
  
  -- Azimuth angle (from North, clockwise)
  let azimuthY = -sin hourAngle in
  let azimuthX = tan declination * cos latitude - sin latitude * cos hourAngle in
  let azimuth = atan2 azimuthY azimuthX in
  
  -- Convert to Cartesian unit vector
  -- x: East, y: North, z: Up
  [ cos elevation * sin azimuth,   -- East
    cos elevation * cos azimuth,   -- North  
    sin elevation ]                -- Up

-- Calculate sun position in Earth-Centered Earth-Fixed (ECEF) coordinates
sunPosition : Point -> R3
sunPosition p =
  let julianDay = timeToJulianDay p in
  let sunAngles = calculateSunAngles julianDay p in
  let declination = sunAngles ! 0 in
  let hourAngle = sunAngles ! 1 in
  let latitude = p ! lat * pi / 180 in
  -- Convert to local topocentric coordinates
  sunToTopocentric latitude declination hourAngle


calculateIncidentPower : InputTensor -> HeatFlux
calculateIncidentPower input =
  let flux = extractSolarFlux input in
  sum (foreach i . 
    let heliostat = field_layout ! i in
    let efficiency = opticalEfficiency heliostat (sunPosition input) receiver_position mirror_material in
    (sum flux) * efficiency * heliostatArea)


extractAmbientTemp inp = 293

computeReceiverState : InputTensor -> ReceiverState
computeReceiverState input =
  let incident = calculateIncidentPower input in
  let wind = extractWindflow input in
  let ambientT = extractAmbientTemp input in
  computeReceiverEnergyBalance incident wind ambientT receiver_material


@property
energyConservation : Bool
energyConservation = forall input .
  let incidentPower = calculateIncidentPower input in
  let receiverState = computeReceiverState input in
  let totalOut = receiverState ! emitted_power_idx + 
                 receiverState ! conducted_power_idx in
  abs (receiverState ! absorbed_power_idx - totalOut) < tolerance


thermalToElectricEfficiency : ReceiverState -> Rat
thermalToElectricEfficiency state =
  0.35  -- Typical steam turbine efficiency


-- Carnot efficiency limit
@property
carnotLimit : Bool  
carnotLimit = forall receiverState ambient .
  let hot = receiverState ! receiver_temp_idx in
  let efficiency = thermalToElectricEfficiency receiverState in
  efficiency <= 1 - ambient / hot


-- Structural safety
@property
structuralIntegrity : Bool
structuralIntegrity = forall input .
  let yieldStrength = 250000000 in  -- Pa for steel
  let safetyFactor = 0.5 in  -- 2x safety margin
  let wind = extractWindflow input in
  let rotation = heliostatController input in
  forall i . 
    let load = computeWindLoad wind rotation heliostatArea in
    let stress = computeStructuralLoads load (field_layout ! i) in
    stress < yieldStrength * safetyFactor

-- @parameter(default=)  -- radians)
alignmentTolerance : Rat
alignmentTolerance = 0.001

computeIdealNormal : Point -> R3 -> R3 -> R3
computeIdealNormal heliostat sunDir receiver =
  let toSun = normalize sunDir in
  let toRec = normalize (receiver - heliostat) in
  normalize (toSun + toRec)


-- Optical path verification
@property
opticalAlignment : Bool
opticalAlignment = forall input .
  -- sunPosition returns a unit vector
  let sunDir = sunPosition input in
  forall i .
    let heliostat = field_layout ! i in
    let normal = computeIdealNormal heliostat sunDir receiver_position in
    let actual = extractNormal (heliostatController input) in
    angleDeviation normal actual < alignmentTolerance


-- Master physics validation
@property
physicallyValid : Bool
physicallyValid = 
  energyConservation and
  carnotLimit and
  structuralIntegrity and
  opticalAlignment
