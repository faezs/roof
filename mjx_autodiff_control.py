"""
MuJoCo-MJX Autodifferentiable Control for Heliostats

This module implements advanced control algorithms using JAX autodiff capabilities
for optimal heliostat tracking with physics-based constraints.

Key Features:
- Model Predictive Control (MPC) with autodiff optimization
- Optimal tracking under wind disturbances  
- Real-time trajectory optimization
- Integration with verified safety constraints
- Neural network-based adaptive control
"""

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from typing import Dict, List, Tuple, Callable, NamedTuple
import flax.linen as nn
from flax.training import train_state
import numpy as np
import logging

from mujoco_heliostat_sim import MuJoCoHeliostatSimulator, HeliostatState, HeliostatConfig

logger = logging.getLogger(__name__)

class MPCParams(NamedTuple):
    """Model Predictive Control parameters"""
    horizon: int = 20  # prediction horizon steps
    dt: float = 0.1   # MPC timestep
    Q_tracking: float = 100.0  # tracking error weight
    Q_velocity: float = 1.0    # velocity penalty weight
    R_torque: float = 0.1      # torque penalty weight
    max_iterations: int = 50   # optimization iterations
    learning_rate: float = 0.01

class TrajectoryState(NamedTuple):
    """State for trajectory optimization"""
    positions: jnp.ndarray     # [horizon, n_heliostats, 2] angles
    velocities: jnp.ndarray    # [horizon, n_heliostats, 2] angular velocities  
    torques: jnp.ndarray       # [horizon, n_heliostats, 2] control torques

class AdaptiveController(nn.Module):
    """Neural network for adaptive heliostat control"""
    
    hidden_dims: List[int] = [64, 64, 32]
    
    @nn.compact
    def __call__(self, state_features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict control corrections based on current state
        
        Parameters
        ----------
        state_features : jnp.ndarray
            Feature vector including tracking errors, wind, vibrations
            
        Returns
        -------
        jnp.ndarray
            Control torque corrections [n_heliostats, 2]
        """
        x = state_features
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)  # Smooth activation for control
            
        # Output layer - control corrections
        control_corrections = nn.Dense(2)(x)  # [azimuth, elevation] corrections
        
        # Limit corrections to reasonable range
        control_corrections = jnp.tanh(control_corrections) * 100.0  # ±100 Nm
        
        return control_corrections

class MJXAutodiffController:
    """Advanced controller using JAX autodiff for optimal heliostat control"""
    
    def __init__(self, 
                 simulator: MuJoCoHeliostatSimulator,
                 mpc_params: MPCParams = MPCParams()):
        """
        Initialize autodiff controller
        
        Parameters
        ----------
        simulator : MuJoCoHeliostatSimulator
            Physics simulator
        mpc_params : MPCParams
            MPC configuration parameters
        """
        self.simulator = simulator
        self.mpc_params = mpc_params
        self.num_heliostats = simulator.num_heliostats
        
        # Initialize adaptive neural network controller
        self.adaptive_net = AdaptiveController()
        self.rng_key = jax.random.PRNGKey(42)
        
        # Initialize network parameters
        dummy_features = jnp.zeros(10)  # Feature vector size
        self.net_params = self.adaptive_net.init(self.rng_key, dummy_features)
        
        # Training state for adaptive controller
        self.optimizer = optax.adam(learning_rate=0.001)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.adaptive_net.apply,
            params=self.net_params,
            tx=self.optimizer
        )
        
        # Cost function weights
        self.Q_track = mpc_params.Q_tracking
        self.Q_vel = mpc_params.Q_velocity  
        self.R_torque = mpc_params.R_torque
        
        logger.info("Initialized MJX autodiff controller")
        
    @jax.jit
    def mpc_step(self, 
                 current_state: TrajectoryState,
                 target_trajectory: jnp.ndarray,
                 wind_forecast: jnp.ndarray) -> jnp.ndarray:
        """
        Model Predictive Control step using autodiff optimization
        
        Parameters
        ----------
        current_state : TrajectoryState
            Current heliostat state
        target_trajectory : jnp.ndarray
            Reference trajectory [horizon, n_heliostats, 2]
        wind_forecast : jnp.ndarray
            Wind prediction [horizon, 3]
            
        Returns
        -------
        jnp.ndarray
            Optimal control torques [n_heliostats, 2]
        """
        
        # Define cost function for trajectory optimization
        def trajectory_cost(torque_sequence: jnp.ndarray) -> float:
            """Cost function for MPC optimization"""
            
            total_cost = 0.0
            state = current_state
            
            for t in range(self.mpc_params.horizon):
                # Predict next state using simplified dynamics
                next_state = self._predict_next_state(
                    state, torque_sequence[t], wind_forecast[t]
                )
                
                # Tracking error cost
                position_error = next_state.positions[0] - target_trajectory[t]
                tracking_cost = self.Q_track * jnp.sum(position_error**2)
                
                # Velocity smoothness cost
                velocity_cost = self.Q_vel * jnp.sum(next_state.velocities[0]**2)
                
                # Control effort cost
                torque_cost = self.R_torque * jnp.sum(torque_sequence[t]**2)
                
                total_cost += tracking_cost + velocity_cost + torque_cost
                
                state = next_state
                
            return total_cost
            
        # Initialize control sequence
        init_torques = jnp.zeros((self.mpc_params.horizon, self.num_heliostats, 2))
        
        # Optimize using gradient descent
        torque_sequence = init_torques
        
        for _ in range(self.mpc_params.max_iterations):
            cost_grad = grad(trajectory_cost)(torque_sequence)
            torque_sequence = torque_sequence - self.mpc_params.learning_rate * cost_grad
            
            # Apply torque limits
            max_torque = self.simulator.heliostat_configs[0].max_torque
            torque_sequence = jnp.clip(torque_sequence, -max_torque, max_torque)
        
        # Return first control action
        return torque_sequence[0]
    
    @jax.jit 
    def _predict_next_state(self, 
                           state: TrajectoryState,
                           torques: jnp.ndarray, 
                           wind: jnp.ndarray) -> TrajectoryState:
        """Simplified physics prediction for MPC"""
        
        # Simplified heliostat dynamics: τ = J*α + b*ω + wind_torque
        # where J=inertia, α=acceleration, b=damping, ω=velocity
        
        inertia = 500.0  # kg⋅m² (typical for 2m x 1m mirror)
        damping = 50.0   # N⋅m⋅s/rad
        
        # Wind torque estimation (simplified)
        wind_speed = jnp.linalg.norm(wind)
        wind_torque_scale = 0.1 * wind_speed**2  # Quadratic with wind speed
        wind_torques = jnp.ones((self.num_heliostats, 2)) * wind_torque_scale
        
        # Equation of motion: α = (τ - b*ω - τ_wind) / J
        net_torques = torques - damping * state.velocities[0] - wind_torques
        accelerations = net_torques / inertia
        
        # Integrate using Euler method
        dt = self.mpc_params.dt
        new_velocities = state.velocities[0] + accelerations * dt
        new_positions = state.positions[0] + new_velocities * dt
        
        # Create new trajectory state (shift and update first step)
        new_positions_traj = jnp.roll(state.positions, -1, axis=0)
        new_positions_traj = new_positions_traj.at[0].set(new_positions)
        
        new_velocities_traj = jnp.roll(state.velocities, -1, axis=0)  
        new_velocities_traj = new_velocities_traj.at[0].set(new_velocities)
        
        new_torques_traj = jnp.roll(state.torques, -1, axis=0)
        new_torques_traj = new_torques_traj.at[0].set(torques)
        
        return TrajectoryState(
            positions=new_positions_traj,
            velocities=new_velocities_traj,
            torques=new_torques_traj
        )
        
    def extract_state_features(self, 
                              heliostat_states: List[HeliostatState],
                              target_angles: jnp.ndarray,
                              wind_velocity: jnp.ndarray) -> jnp.ndarray:
        """Extract features for adaptive controller"""
        
        features = []
        
        for i, state in enumerate(heliostat_states):
            # Tracking errors
            az_error = target_angles[i, 0] - state.azimuth
            el_error = target_angles[i, 1] - state.elevation
            
            # Velocities
            az_vel = state.azimuth_velocity
            el_vel = state.elevation_velocity
            
            # Wind effects
            wind_speed = jnp.linalg.norm(wind_velocity)
            wind_direction = wind_velocity / (wind_speed + 1e-6)
            
            heliostat_features = jnp.array([
                az_error, el_error,
                az_vel, el_vel, 
                wind_speed,
                wind_direction[0], wind_direction[1], wind_direction[2],
                state.torques[0], state.torques[1]
            ])
            
            features.append(heliostat_features)
            
        return jnp.array(features)
        
    def adaptive_control_step(self,
                             heliostat_states: List[HeliostatState],
                             target_angles: jnp.ndarray,
                             wind_velocity: jnp.ndarray,
                             baseline_torques: jnp.ndarray) -> Tuple[jnp.ndarray, train_state.TrainState]:
        """
        Adaptive control using neural network
        
        Parameters
        ----------
        heliostat_states : List[HeliostatState]
            Current heliostat states
        target_angles : jnp.ndarray
            Target angles [n_heliostats, 2]
        wind_velocity : jnp.ndarray
            Current wind velocity [3]
        baseline_torques : jnp.ndarray
            Baseline control torques [n_heliostats, 2]
            
        Returns
        -------
        Tuple[jnp.ndarray, train_state.TrainState]
            Enhanced control torques and updated training state
        """
        
        # Extract features for each heliostat
        state_features = self.extract_state_features(
            heliostat_states, target_angles, wind_velocity
        )
        
        # Get control corrections from adaptive network
        total_corrections = jnp.zeros((self.num_heliostats, 2))
        
        for i in range(self.num_heliostats):
            correction = self.adaptive_net.apply(
                self.train_state.params, state_features[i]
            )
            total_corrections = total_corrections.at[i].set(correction)
        
        # Combine baseline control with adaptive corrections
        enhanced_torques = baseline_torques + total_corrections
        
        # Apply safety limits
        for i, config in enumerate(self.simulator.heliostat_configs):
            enhanced_torques = enhanced_torques.at[i].set(
                jnp.clip(enhanced_torques[i], -config.max_torque, config.max_torque)
            )
            
        return enhanced_torques, self.train_state
        
    def train_adaptive_controller(self,
                                 training_data: List[Dict],
                                 epochs: int = 100) -> train_state.TrainState:
        """
        Train the adaptive controller using historical data
        
        Parameters
        ----------
        training_data : List[Dict]
            Historical state-action pairs
        epochs : int
            Number of training epochs
            
        Returns
        -------
        train_state.TrainState
            Updated training state
        """
        
        def loss_fn(params, batch):
            """Training loss function"""
            features, target_corrections = batch
            
            predicted_corrections = self.adaptive_net.apply(params, features)
            
            # MSE loss
            mse_loss = jnp.mean((predicted_corrections - target_corrections)**2)
            
            # L2 regularization
            l2_loss = 0.001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            
            return mse_loss + l2_loss
            
        @jax.jit
        def train_step(state, batch):
            """Single training step"""
            loss_value, grads = jax.value_and_grad(loss_fn)(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, loss_value
            
        # Prepare training batches (simplified)
        logger.info(f"Training adaptive controller for {epochs} epochs")
        
        for epoch in range(epochs):
            # In practice, would batch the training data
            epoch_loss = 0.0
            
            for data_point in training_data:
                features = data_point['features']
                target_corrections = data_point['target_corrections']
                batch = (features, target_corrections)
                
                self.train_state, loss = train_step(self.train_state, batch)
                epoch_loss += loss
                
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(training_data)
                logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.6f}")
                
        return self.train_state
        
    @jax.jit
    def safety_filter(self, 
                     proposed_torques: jnp.ndarray,
                     current_states: List[HeliostatState],
                     wind_speed: float) -> jnp.ndarray:
        """
        Safety filter for control commands using autodiff constraints
        
        Parameters
        ---------- 
        proposed_torques : jnp.ndarray
            Proposed control torques [n_heliostats, 2]
        current_states : List[HeliostatState] 
            Current heliostat states
        wind_speed : float
            Current wind speed
            
        Returns
        -------
        jnp.ndarray
            Safety-filtered torques
        """
        
        # Define safety constraints
        def safety_cost(torques):
            """Cost function penalizing unsafe actions"""
            
            safety_cost = 0.0
            
            # Wind speed constraint
            if wind_speed > 20.0:  # High wind condition
                # Penalize large torques in high wind
                wind_penalty = 1000.0 * jnp.sum(torques**2) * (wind_speed / 20.0)**2
                safety_cost += wind_penalty
                
            # Velocity limit constraint
            for i, state in enumerate(current_states):
                max_vel = 10.0  # deg/s
                
                if abs(state.azimuth_velocity) > max_vel:
                    vel_penalty = 1000.0 * (abs(state.azimuth_velocity) - max_vel)**2
                    safety_cost += vel_penalty
                    
                if abs(state.elevation_velocity) > max_vel:
                    vel_penalty = 1000.0 * (abs(state.elevation_velocity) - max_vel)**2  
                    safety_cost += vel_penalty
                    
            return safety_cost
            
        # If proposed torques are safe, return them
        if safety_cost(proposed_torques) < 1.0:
            return proposed_torques
            
        # Otherwise, optimize for safe torques
        safe_torques = proposed_torques.copy()
        
        for _ in range(10):  # Quick optimization
            grad_safety = grad(safety_cost)(safe_torques)
            safe_torques = safe_torques - 0.1 * grad_safety
            
            # Apply hard limits
            for i, config in enumerate(self.simulator.heliostat_configs):
                safe_torques = safe_torques.at[i].set(
                    jnp.clip(safe_torques[i], -config.max_torque, config.max_torque)
                )
                
        return safe_torques
        
    def generate_optimal_trajectory(self,
                                  current_angles: jnp.ndarray,
                                  target_angles: jnp.ndarray,
                                  wind_forecast: jnp.ndarray,
                                  trajectory_time: float = 10.0) -> jnp.ndarray:
        """
        Generate optimal trajectory using autodiff optimization
        
        Parameters
        ----------
        current_angles : jnp.ndarray
            Current heliostat angles [n_heliostats, 2]
        target_angles : jnp.ndarray  
            Target heliostat angles [n_heliostats, 2]
        wind_forecast : jnp.ndarray
            Wind forecast [time_steps, 3]
        trajectory_time : float
            Trajectory duration in seconds
            
        Returns
        -------
        jnp.ndarray
            Optimal trajectory [time_steps, n_heliostats, 2]
        """
        
        time_steps = int(trajectory_time / self.mpc_params.dt)
        
        # Define trajectory optimization problem
        def trajectory_cost(trajectory_params):
            """Cost function for trajectory optimization"""
            
            # Reconstruct trajectory from parameters
            trajectory = self._params_to_trajectory(
                trajectory_params, current_angles, target_angles, time_steps
            )
            
            total_cost = 0.0
            
            for t in range(time_steps):
                # Tracking error
                error = trajectory[t] - target_angles
                tracking_cost = jnp.sum(error**2)
                
                # Smoothness cost
                if t > 0:
                    velocity = (trajectory[t] - trajectory[t-1]) / self.mpc_params.dt
                    smoothness_cost = 0.1 * jnp.sum(velocity**2)
                    total_cost += smoothness_cost
                    
                # Wind disturbance compensation
                wind_compensation_cost = 0.01 * jnp.sum(
                    trajectory[t]**2 * jnp.linalg.norm(wind_forecast[t])
                )
                
                total_cost += tracking_cost + wind_compensation_cost
                
            return total_cost
            
        # Initialize trajectory parameters
        init_params = jnp.zeros((time_steps, self.num_heliostats, 2))
        
        # Optimize trajectory
        optimal_params = init_params
        learning_rate = 0.05
        
        for iteration in range(100):
            traj_grad = grad(trajectory_cost)(optimal_params)
            optimal_params = optimal_params - learning_rate * traj_grad
            
            if iteration % 20 == 0:
                cost = trajectory_cost(optimal_params)
                logger.debug(f"Trajectory optimization iter {iteration}: cost = {cost:.4f}")
                
        # Convert parameters back to trajectory
        optimal_trajectory = self._params_to_trajectory(
            optimal_params, current_angles, target_angles, time_steps
        )
        
        return optimal_trajectory
        
    def _params_to_trajectory(self,
                            params: jnp.ndarray,
                            start_angles: jnp.ndarray,
                            end_angles: jnp.ndarray,
                            time_steps: int) -> jnp.ndarray:
        """Convert optimization parameters to trajectory"""
        
        # Use cubic spline interpolation with parameters as control points
        trajectory = jnp.zeros((time_steps, self.num_heliostats, 2))
        
        for t in range(time_steps):
            alpha = t / (time_steps - 1)  # [0, 1]
            
            # Cubic interpolation with parameter influence
            base_traj = start_angles + alpha * (end_angles - start_angles)
            param_influence = params[t] * alpha * (1 - alpha) * 4  # Peak at alpha=0.5
            
            trajectory = trajectory.at[t].set(base_traj + param_influence)
            
        return trajectory


def create_wind_forecast(duration: float, timestep: float, base_wind: jnp.ndarray) -> jnp.ndarray:
    """Generate realistic wind forecast for MPC"""
    
    steps = int(duration / timestep)
    forecast = jnp.zeros((steps, 3))
    
    time_vector = jnp.arange(steps) * timestep
    
    # Add temporal variations
    for i in range(steps):
        t = time_vector[i]
        
        # Sinusoidal variations at different frequencies
        variation = (0.3 * jnp.sin(0.1 * t) + 
                    0.2 * jnp.sin(0.05 * t) + 
                    0.1 * jnp.sin(0.02 * t))
        
        current_wind = base_wind * (1 + variation)
        forecast = forecast.at[i].set(current_wind)
        
    return forecast


def main():
    """Demo of MJX autodiff control"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Import heliostat simulator setup  
    from mujoco_heliostat_sim import create_demo_heliostat_field, WindDisturbance
    
    # Create heliostat field
    heliostat_configs = create_demo_heliostat_field()
    
    # Wind configuration
    wind_config = WindDisturbance(
        mean_velocity=jnp.array([10.0, 3.0, 0.0]),
        turbulence_intensity=0.2,
        gust_factor=1.5,
        coherence_length=40.0
    )
    
    # Initialize simulator
    simulator = MuJoCoHeliostatSimulator(
        heliostat_configs=heliostat_configs,
        wind_config=wind_config,
        timestep=0.01
    )
    
    # Initialize autodiff controller
    mpc_params = MPCParams(
        horizon=15,
        dt=0.1,
        Q_tracking=200.0,
        Q_velocity=2.0,
        R_torque=0.05,
        max_iterations=30
    )
    
    controller = MJXAutodiffController(simulator, mpc_params)
    
    # Define target tracking scenario
    current_angles = jnp.zeros((9, 2))  # 9 heliostats, flat initially
    target_angles = jnp.array([[45.0, 60.0]] * 9)  # Track towards southeast, elevated
    
    # Generate wind forecast
    wind_forecast = create_wind_forecast(
        duration=10.0, 
        timestep=0.1, 
        base_wind=wind_config.mean_velocity
    )
    
    logger.info("Running MPC trajectory optimization...")
    
    # Generate optimal trajectory
    optimal_traj = controller.generate_optimal_trajectory(
        current_angles=current_angles,
        target_angles=target_angles, 
        wind_forecast=wind_forecast,
        trajectory_time=10.0
    )
    
    logger.info(f"Generated optimal trajectory: {optimal_traj.shape}")
    logger.info(f"Initial angles: {optimal_traj[0]}")
    logger.info(f"Final angles: {optimal_traj[-1]}")
    
    # Test MPC control step
    initial_state = TrajectoryState(
        positions=jnp.tile(current_angles[None, :, :], (20, 1, 1)),
        velocities=jnp.zeros((20, 9, 2)),
        torques=jnp.zeros((20, 9, 2))
    )
    
    optimal_torques = controller.mpc_step(
        current_state=initial_state,
        target_trajectory=optimal_traj[:20],  # First 20 steps
        wind_forecast=wind_forecast[:20]
    )
    
    logger.info(f"Optimal control torques: {optimal_torques}")
    logger.info("MJX autodiff control demo completed successfully!")


if __name__ == "__main__":
    main()