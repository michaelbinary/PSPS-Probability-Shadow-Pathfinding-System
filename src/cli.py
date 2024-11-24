#!/usr/bin/env python3
import click
from rich.console import Console
import numpy as np
from pathlib import Path
from .simulation import ProbabilityShadowGrid
from .visualization.animator import SimulationAnimator

console = Console()


@click.group()
def cli():
    """Probability Shadow Pathfinding System (PSPS) Command Line Interface"""
    pass


@cli.command()
@click.option('--space-x', default=50, help='X dimension of the space')
@click.option('--space-y', default=50, help='Y dimension of space')
@click.option('--space-z', default=30, help='Z dimension of space')
@click.option('--num-drones', default=6, help='Number of drones to simulate')
@click.option('--prediction-steps', default=20, help='Number of prediction steps')
@click.option('--uncertainty-rate', default=0.08, help='Uncertainty growth rate')
@click.option('--collision-threshold', default=0.25, help='Collision risk threshold')
@click.option('--output-dir', default=None, help='Output directory for results')
@click.option('--frames', default=500, help='Number of simulation frames')
def run(space_x, space_y, space_z, num_drones, prediction_steps,
        uncertainty_rate, collision_threshold, output_dir, frames):
    """Run a probability shadow pathfinding simulation"""
    try:
        console.print("[bold blue]Starting PSPS Simulation[/bold blue]")

        # Configure space and grid
        space_size = (space_x, space_y, space_z)
        grid = ProbabilityShadowGrid(space_size)

        # Configure parameters
        grid.prediction_steps = prediction_steps
        grid.uncertainty_growth_rate = uncertainty_rate
        grid.collision_threshold = collision_threshold

        # Initialize drones
        max_x, max_y, max_z = space_size
        for i in range(num_drones):
            start = np.array([
                np.random.uniform(0.1, 0.3) * max_x,
                np.random.uniform(0.1, 0.9) * max_y,
                np.random.uniform(0.2, 0.8) * max_z
            ])
            goal = np.array([
                np.random.uniform(0.7, 0.9) * max_x,
                np.random.uniform(0.1, 0.9) * max_y,
                np.random.uniform(0.2, 0.8) * max_z
            ])
            grid.initialize_drone(f"drone_{i}", start, goal)

        # Create animator and run simulation
        animator = SimulationAnimator(grid, output_dir)
        console.print("[yellow]Running simulation...[/yellow]")

        for frame in range(frames):
            if frame % 50 == 0:
                console.print(f"Frame {frame}/{frames}")
            grid.update(0.1)
            animator.update(frame)

        animator.save_animation()
        console.print("[green]Simulation completed successfully![/green]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise


@cli.command()
def info():
    """Display information about the simulation system"""
    console.print("""
[bold blue]Probability Shadow Pathfinding System (PSPS)[/bold blue]

[yellow]Key Features:[/yellow]
- Probability shadow-based collision prediction
- Dynamic uncertainty propagation
- Adaptive path rerouting
- Real-time risk assessment

[yellow]Parameters:[/yellow]
- Space Dimensions: 3D space size (X, Y, Z)
- Prediction Steps: Number of future positions to predict
- Uncertainty Rate: Growth rate of position uncertainty
- Collision Threshold: Risk threshold for rerouting

[yellow]Example Usage:[/yellow]
  # Run basic simulation
  psps run

  # Run with custom parameters
  psps run --space-x 60 --space-y 60 --space-z 40 --num-drones 8

  # Adjust risk parameters
  psps run --uncertainty-rate 0.1 --collision-threshold 0.3
    """)


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()