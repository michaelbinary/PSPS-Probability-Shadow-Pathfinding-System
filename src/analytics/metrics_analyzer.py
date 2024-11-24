import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from ..core.metrics import SimulationMetrics
import json


class MetricsAnalyzer:
    """Analyzes and saves simulation metrics"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics_history: List[SimulationMetrics]):
        """Save metrics in various formats with proper type conversion"""
        metrics_data = [
            {
                'timestamp': float(metric.timestamp),
                'average_collision_risk': float(metric.average_collision_risk),
                'max_collision_risk': float(metric.max_collision_risk),
                **{f'distance_{k}': float(v) for k, v in metric.total_distance_traveled.items()},
                **{f'velocity_{k}': float(v) for k, v in metric.average_velocity.items()},
                **{f'rerouting_{k}': int(v) for k, v in metric.rerouting_events.items()},
                **{f'completion_{k}': float(v) for k, v in metric.completion_percentage.items()}
            }
            for metric in metrics_history
        ]

        # Save as CSV
        df = pd.DataFrame(metrics_data)
        df.to_csv(self.output_dir / 'simulation_metrics.csv', index=False)

        # Save as JSON
        with open(self.output_dir / 'simulation_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Generate summary statistics
        summary_stats = self._generate_summary_stats(df)
        with open(self.output_dir / 'simulation_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)

    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from metrics data"""
        return {
            'total_simulation_time': float(df['timestamp'].max()),
            'average_collision_risk': float(df['average_collision_risk'].mean()),
            'max_collision_risk_observed': float(df['max_collision_risk'].max()),
            'final_completion_percentage': {
                col.replace('completion_', ''): float(df[col].iloc[-1])
                for col in df.columns if col.startswith('completion_')
            },
            'total_rerouting_events': {
                col.replace('rerouting_', ''): int(df[col].iloc[-1])
                for col in df.columns if col.startswith('rerouting_')
            },
            'average_velocities': {
                col.replace('velocity_', ''): float(df[col].mean())
                for col in df.columns if col.startswith('velocity_')
            }
        }