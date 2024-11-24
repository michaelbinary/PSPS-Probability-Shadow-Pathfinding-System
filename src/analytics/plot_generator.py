import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class PlotGenerator:
    """Generates visualization plots from simulation metrics"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_plots(self, df: pd.DataFrame):
        """Generate all analysis plots"""
        self._plot_collision_risks(df)
        self._plot_completion_progress(df)
        self._plot_velocities(df)
        self._plot_rerouting_events(df)

    def _plot_collision_risks(self, df: pd.DataFrame):
        """Plot collision risk trends"""
        plt.figure(figsize=(12, 8))
        plt.plot(df['timestamp'], df['average_collision_risk'], label='Average Risk')
        plt.plot(df['timestamp'], df['max_collision_risk'], label='Max Risk')
        plt.xlabel('Time (s)')
        plt.ylabel('Collision Risk')
        plt.title('Collision Risk Over Time')
        plt.legend()
        plt.savefig(self.output_dir / 'collision_risk_plot.png')
        plt.close()

    def _plot_completion_progress(self, df: pd.DataFrame):
        """Plot mission completion progress"""
        completion_cols = [col for col in df.columns if col.startswith('completion_')]
        plt.figure(figsize=(12, 8))
        for col in completion_cols:
            plt.plot(df['timestamp'], df[col],
                     label=col.replace('completion_', 'Drone '))
        plt.xlabel('Time (s)')
        plt.ylabel('Completion Percentage')
        plt.title('Mission Completion Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'completion_progress_plot.png')
        plt.close()

    def _plot_velocities(self, df: pd.DataFrame):
        """Plot drone velocities"""
        velocity_cols = [col for col in df.columns if col.startswith('velocity_')]
        plt.figure(figsize=(12, 8))
        for col in velocity_cols:
            plt.plot(df['timestamp'], df[col],
                     label=col.replace('velocity_', 'Drone '))
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Drone Velocities Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'velocities_plot.png')
        plt.close()

    def _plot_rerouting_events(self, df: pd.DataFrame):
        """Plot cumulative rerouting events"""
        rerouting_cols = [col for col in df.columns if col.startswith('rerouting_')]
        plt.figure(figsize=(12, 8))
        for col in rerouting_cols:
            plt.plot(df['timestamp'], df[col],
                     label=col.replace('rerouting_', 'Drone '))
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Rerouting Events')
        plt.title('Cumulative Rerouting Events')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'rerouting_events_plot.png')
        plt.close()