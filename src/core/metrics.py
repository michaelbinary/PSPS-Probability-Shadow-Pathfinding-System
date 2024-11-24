from dataclasses import dataclass
from typing import Dict
from datetime import datetime


@dataclass
class SimulationMetrics:
    """Store metrics for each simulation timestep"""
    timestamp: float
    average_collision_risk: float
    max_collision_risk: float
    total_distance_traveled: Dict[str, float]
    average_velocity: Dict[str, float]
    rerouting_events: Dict[str, int]
    completion_percentage: Dict[str, float]

    @classmethod
    def create_from_state(cls, timestamp: float, drones: Dict) -> 'SimulationMetrics':
        """Create metrics from current simulation state"""
        collision_risks = [drone.collision_risk for drone in drones.values()]

        total_distances = {}
        average_velocities = {}
        completion_percentages = {}

        for drone_id, drone in drones.items():
            # Calculate total distance traveled
            path = drone.path_history
            total_distances[drone_id] = sum(
                np.linalg.norm(path[i + 1] - path[i])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0.0

            average_velocities[drone_id] = np.linalg.norm(drone.velocity)

            # Calculate completion percentage
            total_distance = np.linalg.norm(drone.goal - drone.start_position)
            current_distance = np.linalg.norm(drone.goal - drone.position)
            completion_percentages[drone_id] = min(
                100 * (1 - current_distance / total_distance), 100
            ) if total_distance > 0 else 100

        return cls(
            timestamp=timestamp,
            average_collision_risk=np.mean(collision_risks),
            max_collision_risk=np.max(collision_risks),
            total_distance_traveled=total_distances,
            average_velocity=average_velocities,
            rerouting_events={drone_id: drone.rerouting_count for drone_id, drone in drones.items()},
            completion_percentage=completion_percentages
        )