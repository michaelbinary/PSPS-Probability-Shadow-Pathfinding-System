from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from pathlib import Path
from datetime import datetime
from ..core.metrics import SimulationMetrics
from ..analytics.metrics_analyzer import MetricsAnalyzer
from ..analytics.plot_generator import PlotGenerator
from rich.console import Console

console = Console()


class ProbabilityShadowSimulation:
    """Main simulation class handling visualization and execution"""

    def __init__(self, space_size=(50, 50, 30)):
        self.space_size = space_size
        self.grid = ProbabilityShadowGrid(space_size)
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.drone_colors = {}

        # Create output directory
        self.output_dir = Path('simulation_output') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analytics
        self.metrics_analyzer = MetricsAnalyzer(self.output_dir)
        self.plot_generator = PlotGenerator(self.output_dir)

        self.setup_visualization()

    def setup_visualization(self):
        """Set up the 3D visualization environment"""
        self.ax.set_xlim(0, self.space_size[0])
        self.ax.set_ylim(0, self.space_size[1])
        self.ax.set_zlim(0, self.space_size[2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def initialize_scenario(self):
        """Initialize demonstration scenario with crossing paths"""
        num_drones = 6
        for i in range(num_drones):
            self.drone_colors[f'drone_{i}'] = colorsys.hsv_to_rgb(i / num_drones, 0.8, 0.9)

        max_x, max_y, max_z = self.space_size
        scenarios = [
            ("drone_0", np.array([0.2 * max_x, 0.2 * max_y, 0.5 * max_z]),
             np.array([0.8 * max_x, 0.8 * max_y, 0.5 * max_z])),
            ("drone_1", np.array([0.8 * max_x, 0.2 * max_y, 0.3 * max_z]),
             np.array([0.2 * max_x, 0.8 * max_y, 0.7 * max_z])),
            ("drone_2", np.array([0.5 * max_x, 0.1 * max_y, 0.2 * max_z]),
             np.array([0.5 * max_x, 0.9 * max_y, 0.8 * max_z])),
            ("drone_3", np.array([0.1 * max_x, 0.5 * max_y, 0.6 * max_z]),
             np.array([0.9 * max_x, 0.5 * max_y, 0.4 * max_z])),
            ("drone_4", np.array([0.9 * max_x, 0.9 * max_y, 0.3 * max_z]),
             np.array([0.1 * max_x, 0.1 * max_y, 0.7 * max_z])),
            ("drone_5", np.array([0.3 * max_x, 0.8 * max_y, 0.8 * max_z]),
             np.array([0.7 * max_x, 0.2 * max_y, 0.2 * max_z]))
        ]

        for drone_id, start, goal in scenarios:
            self.grid.initialize_drone(drone_id, start, goal)
            console.print(f"[green]Added {drone_id}[/green]")

    def update(self, frame):
        """Update visualization frame"""
        self.ax.clear()
        self.setup_visualization()

        self.grid.update(0.1)

        for drone_id, drone in self.grid.drones.items():
            color = self.drone_colors[drone_id]

            # Draw probability shadow
            shadow_positions = np.array([p[0] for p in drone.shadow_points])
            shadow_probs = np.array([p[1] for p in drone.shadow_points])
            sizes = shadow_probs * 50

            self.ax.scatter(
                shadow_positions[:, 0],
                shadow_positions[:, 1],
                shadow_positions[:, 2],
                c=[color],
                s=sizes,
                alpha=shadow_probs * 0.3
            )

            # Draw drone
            self.ax.scatter(
                *drone.position,
                color=color,
                s=100,
                label=f'Drone {drone_id} (Risk: {drone.collision_risk:.2f})'
            )

            # Draw goal
            self.ax.scatter(
                *drone.goal,
                color=color,
                marker='*',
                s=200,
                alpha=0.5
            )

            # Draw path history
            if len(drone.path_history) > 1:
                path = np.array(drone.path_history)
                self.ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    '-',
                    color=color,
                    alpha=0.5
                )

            # Draw velocity vector if moving
            if np.any(drone.velocity):
                self.ax.quiver(
                    *drone.position,
                    *drone.velocity,
                    color=color,
                    length=2.0,
                    alpha=0.6
                )

        # Add legend and title
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Update title with current metrics
        metrics = self.grid.metrics_history[-1] if self.grid.metrics_history else None
        if metrics:
            title = (f'Time Step: {frame}\n'
                     f'Avg Risk: {metrics.average_collision_risk:.3f}\n'
                     f'Max Risk: {metrics.max_collision_risk:.3f}')
        else:
            title = f'Time Step: {frame}'

        self.ax.set_title(title)

        return self.ax.get_children()

    def run(self, frames=500):
        """Run the simulation"""
        console.print("[bold blue]Starting Enhanced Probability Shadow Pathfinding Simulation...[/bold blue]")
        self.initialize_scenario()

        # Create animation
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=50,
            blit=False,
            repeat=False
        )

        # Save animation
        console.print("[yellow]Saving animation...[/yellow]")
        writer = PillowWriter(fps=20)
        anim.save(self.output_dir / 'simulation.gif', writer=writer)

        # Save analytics
        console.print("[yellow]Generating analytics...[/yellow]")
        self.metrics_analyzer.save_metrics(self.grid.metrics_history)

        # Generate plots
        console.print("[yellow]Generating visualization plots...[/yellow]")
        self.plot_generator.generate_plots(
            pd.DataFrame([m.__dict__ for m in self.grid.metrics_history])
        )

        # Print completion status
        final_metrics = self.grid.metrics_history[-1]
        console.print("\n[bold green]Simulation complete![/bold green]")
        console.print("\n[bold blue]Final Statistics:[/bold blue]")

        for drone_id in self.grid.drones:
            console.print(f"\n[yellow]Drone {drone_id}:[/yellow]")
            console.print(f"  Completion: {final_metrics.completion_percentage[drone_id]:.1f}%")
            console.print(f"  Distance: {final_metrics.total_distance_traveled[drone_id]:.1f}")
            console.print(f"  Rerouting Events: {final_metrics.rerouting_events[drone_id]}")

        console.print(f"\nOutput saved to: {self.output_dir}")
        plt.show()