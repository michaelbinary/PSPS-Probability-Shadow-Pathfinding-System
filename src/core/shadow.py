import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Tuple, Optional


class ProbabilityShadow:
    """Handles probability shadow calculations and risk assessment"""

    def __init__(self, prediction_steps: int = 20, uncertainty_growth_rate: float = 0.08,
                 time_horizon: float = 6.0):
        self.prediction_steps = prediction_steps
        self.uncertainty_growth_rate = uncertainty_growth_rate
        self.time_horizon = time_horizon

    def generate_shadow_points(self, position: np.ndarray, velocity: np.ndarray,
                               goal: np.ndarray, covariance: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Generate probability shadow points along predicted path"""
        direction = goal - position
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance

        time_steps = np.linspace(0, self.time_horizon, self.prediction_steps)
        predicted_positions = position + np.outer(time_steps, direction * np.linalg.norm(velocity))

        shadow_points = []
        for step, pred_pos in enumerate(predicted_positions):
            time_factor = 1 + self.uncertainty_growth_rate * time_steps[step]
            step_covariance = covariance * time_factor

            num_points = 5 + step * 2
            points = np.random.multivariate_normal(pred_pos, step_covariance, num_points)

            mvn = multivariate_normal(pred_pos, step_covariance)
            probs = mvn.pdf(points)
            probs = probs / np.max(probs) if np.max(probs) > 0 else probs

            shadow_points.extend([(point, prob) for point, prob in zip(points, probs)])

        return shadow_points

    def calculate_collision_risk(self, shadow1: List[Tuple[np.ndarray, float]],
                                 shadow2: List[Tuple[np.ndarray, float]], safe_distance: float) -> float:
        """Calculate collision risk between two probability shadows"""
        max_prob = 0.0

        pos1 = np.array([p[0] for p in shadow1])
        prob1 = np.array([p[1] for p in shadow1])
        pos2 = np.array([p[0] for p in shadow2])
        prob2 = np.array([p[1] for p in shadow2])

        for i in range(len(pos1)):
            distances = np.linalg.norm(pos1[i] - pos2, axis=1)
            mask = distances < safe_distance
            if any(mask):
                collision_probs = prob1[i] * prob2[mask] * (1 - distances[mask] / safe_distance)
                max_prob = max(max_prob, np.max(collision_probs))

        return max_prob