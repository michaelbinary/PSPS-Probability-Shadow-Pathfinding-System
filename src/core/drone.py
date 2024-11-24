from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class DroneState:
    """Represents the state of a single drone"""
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    uncertainty: float
    covariance: np.ndarray
    shadow_points: List[Tuple[np.ndarray, float]]
    path_history: List[np.ndarray]
    collision_risk: float
    rerouting: bool
    start_position: np.ndarray
    rerouting_count: int
    original_goal: np.ndarray
    rerouting_steps: int

    @classmethod
    def initialize(cls, start: np.ndarray, goal: np.ndarray, initial_uncertainty: float = 0.1):
        """Initialize a new drone state"""
        initial_covariance = np.eye(3) * initial_uncertainty

        return cls(
            position=start.copy(),
            velocity=np.zeros(3),
            goal=goal.copy(),
            uncertainty=initial_uncertainty,
            covariance=initial_covariance,
            shadow_points=[],
            path_history=[start.copy()],
            collision_risk=0.0,
            rerouting=False,
            start_position=start.copy(),
            rerouting_count=0,
            original_goal=goal.copy(),
            rerouting_steps=0
        )

    def update_position(self, new_position: np.ndarray):
        """Update drone position and path history"""
        self.position = new_position.copy()
        self.path_history.append(new_position.copy())

    def start_rerouting(self, new_goal: np.ndarray):
        """Initiate rerouting procedure"""
        self.rerouting = True
        self.rerouting_steps = 0
        self.goal = new_goal.copy()
        self.rerouting_count += 1

    def stop_rerouting(self):
        """End rerouting and return to original goal"""
        self.rerouting = False
        self.rerouting_steps = 0
        self.goal = self.original_goal.copy()

    def update_shadow(self, shadow_points: List[Tuple[np.ndarray, float]]):
        """Update probability shadow points"""
        self.shadow_points = shadow_points