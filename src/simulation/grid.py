from typing import Dict, Tuple, Optional
import numpy as np
from ..core.drone import DroneState
from ..core.shadow import ProbabilityShadow
from ..core.metrics import SimulationMetrics
from rich.console import Console

console = Console()


class ProbabilityShadowGrid:
    """Main simulation grid managing drone interactions and pathfinding"""

    def __init__(self, space_size: Tuple[float, float, float]):
        self.space_size = space_size
        self.drones: Dict[str, DroneState] = {}

        # Configuration
        self.prediction_steps = 20
        self.uncertainty_growth_rate = 0.08
        self.collision_threshold = 0.25
        self.safe_distance = 4.0
        self.time_horizon = 6.0
        self.rerouting_timeout = 50

        # Initialize components
        self.shadow_calculator = ProbabilityShadow(
            self.prediction_steps,
            self.uncertainty_growth_rate,
            self.time_horizon
        )

        # Analytics storage
        self.metrics_history: List[SimulationMetrics] = []
        self.start_time = time.time()

    def initialize_drone(self, drone_id: str, start: np.ndarray, goal: np.ndarray):
        """Initialize a new drone in the simulation"""
        drone = DroneState.initialize(start, goal)
        self.drones[drone_id] = drone
        self._update_shadow(drone_id)
        console.print(f"[green]Initialized drone {drone_id}[/green]")

    def _update_shadow(self, drone_id: str):
        """Update probability shadow for a drone"""
        drone = self.drones[drone_id]
        drone.shadow_points = self.shadow_calculator.generate_shadow_points(
            drone.position,
            drone.velocity,
            drone.goal,
            drone.covariance
        )

    def _find_alternative_path(self, drone_id: str) -> Optional[np.ndarray]:
        """Find alternative path to avoid high probability collision areas"""
        drone = self.drones[drone_id]
        best_alternative = None
        lowest_risk = float('inf')

        direction_to_goal = drone.original_goal - drone.position
        distance_to_goal = np.linalg.norm(direction_to_goal)

        if distance_to_goal > 0:
            direction_to_goal = direction_to_goal / distance_to_goal

            # Search in a cone-shaped space
            radii = [5.0, 10.0, 15.0]
            angles = np.linspace(-np.pi / 2, np.pi / 2, 8)

            for radius in radii:
                for angle in angles:
                    rotation = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])

                    alternative_direction = rotation @ direction_to_goal
                    alt_goal = drone.position + alternative_direction * radius
                    alt_goal = np.clip(alt_goal, 0, np.array(self.space_size))

                    # Test alternative
                    orig_goal = drone.goal
                    drone.goal = alt_goal
                    self._update_shadow(drone_id)
                    risk = max(
                        self.shadow_calculator.calculate_collision_risk(
                            drone.shadow_points,
                            other_drone.shadow_points,
                            self.safe_distance
                        )
                        for other_id, other_drone in self.drones.items()
                        if other_id != drone_id
                    )

                    if risk < lowest_risk:
                        lowest_risk = risk
                        best_alternative = alt_goal.copy()

                    # Restore original goal
                    drone.goal = orig_goal
                    self._update_shadow(drone_id)

        return best_alternative if lowest_risk < drone.collision_risk else None

    def update(self, dt: float):
        """Update simulation state"""
        for drone_id, drone in self.drones.items():
            self._update_drone(drone_id, dt)

        self.metrics_history.append(
            SimulationMetrics.create_from_state(
                time.time() - self.start_time,
                self.drones
            )
        )

    def _update_drone(self, drone_id: str, dt: float):
        """Update individual drone state"""
        drone = self.drones[drone_id]
        distance = np.linalg.norm(drone.original_goal - drone.position)

        if distance > 0.1:
            if drone.rerouting:
                drone.rerouting_steps += 1

                if (drone.rerouting_steps > self.rerouting_timeout or
                        drone.collision_risk < self.collision_threshold * 0.5):
                    drone.stop_rerouting()

            # Calculate movement
            direction_to_goal = drone.goal - drone.position
            distance_to_goal = np.linalg.norm(direction_to_goal)

            if distance_to_goal > 0:
                speed = 1.5 + np.random.normal(0, 0.05)
                drone.velocity = direction_to_goal / distance_to_goal * speed
                new_pos = drone.position + drone.velocity * dt
                drone.position = np.clip(new_pos, 0, np.array(self.space_size))
                drone.update_position(drone.position)

        self._update_shadow(drone_id)

        # Update collision risk and check for rerouting
        drone.collision_risk = max(
            self.shadow_calculator.calculate_collision_risk(
                drone.shadow_points,
                other_drone.shadow_points,
                self.safe_distance
            )
            for other_id, other_drone in self.drones.items()
            if other_id != drone_id
        )

        if drone.collision_risk > self.collision_threshold and not drone.rerouting:
            alternative_goal = self._find_alternative_path(drone_id)
            if alternative_goal is not None:
                drone.start_rerouting(alternative_goal)