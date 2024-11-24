#!/usr/bin/env python3
"""
Basic demonstration of the Probability Shadow Pathfinding System.
Creates a scenario with multiple drones navigating through shared space.
"""

from psps.simulation import ProbabilityShadowGrid
from psps.visualization.animator import SimulationAnimator
from rich.console import Console
import numpy as np

console = Console()


def create_crossing_scenario(grid, space_size):
    """Create a scenario with intentionally crossing paths"""
    scenarios = [
        ("drone_0", np.array([0.2 * space_size[0], 0.2 * space_size[1], 0.5 * space_size[2]]),
         np.array([0.8 * space_size[0], 0.8 * space_size[1], 0.5 * space_size[2]])),
        ("drone_1", np.array([0.8 * space_size[0], 0.2 * space_size[1], 0.3 * space_size[2]]),
         np.array([0.2 * space_size[0], 0.8 * space_size[1], 0.7 * space_size[2]])),
        ("drone_2", np.array([0.5 * space_size[0], 0.1 * space_size[1], 0.2 * space_size[2]]),
         np.array([0.5 * space_size[0], 0.9 * space_size[1], 0.8 * space_size[2]])),
        ("drone_3", np.array([0.1 * space_size[0], 0.5 * space_size[1], 0.6 * space_size[2]]),
         np.array([0.9 * space_size[0], 0.5 * space_size[1], 0.4 * space_size[2]])),
        ("drone_4", np.array([0.9 * space_size[0], 0.9 * space_size[1], 0.3 * space_size[2]]),
         np.array([0.1 * space_size[0], 0.1 * space_size[1], 0.7 * space_size[2]])),
        ("drone_5", np.array([0.3 * space_size[0], 0.8 * space_size[1], 0.8 * space_size[2]]),
         np.array([0.7 * space_size[0], 0.2 * space_size[1], 0.2 * space_size[2]]))
    ]

    for drone_id, start, goal in scenarios:
        grid.initialize_drone(drone_id, start, goal)
        console.print(f"[green]Added {drone_id}[/green]")
        console.print(f"  Start: {start.round(2)}")
        console.print(f"  Goal: {goal.round(2)}")


def main():
    """Run the demonstration simulation"""
    console.print("[bold blue]Starting Probability Shadow Pathfinding Demo[/bold blue]")
    console.print("=" * 60)

    # Create simulation environment
    space_size = (50, 50, 30)
    grid = ProbabilityShadowGrid(space_size)
    console.print(f"[cyan]Space dimensions: {space_size}[/cyan]")

    # Create crossing traffic scenario
    create_crossing_scenario(grid, space_size)

    # Initialize visualizer
    animator = SimulationAnimator(grid)
    console.print("\n[bold green]Visualization initialized[/bold green]")

    # Run simulation
    console.print("\n[bold yellow]Running simulation...[/bold yellow]")
    try:
        frames = 500
        for frame in range(frames):
            if frame % 50 == 0:
                console.print(f"Processing frame {frame}/{frames}")

            grid.update(0.1)
            animator.update(frame)

        # Save results
        animator.save_animation()
        console.print("\n[bold green]Simulation completed successfully![/bold green]")

        # Print final statistics
        metrics = grid.metrics_history[-1]
        console.print("\n[bold blue]Final Statistics:[/bold blue]")
        console.print(f"Average Collision Risk: {metrics.average_collision_risk:.3f}")
        console.print(f"Maximum Collision Risk: {metrics.max_collision_risk:.3f}")

        for drone_id in grid.drones:
            console.print(f"\n[yellow]Drone {drone_id} Statistics:[/yellow]")
            console.print(f"  Completion: {metrics.completion_percentage[drone_id]:.1f}%")
            console.print(f"  Total Distance: {metrics.total_distance_traveled[drone_id]:.1f}")
            console.print(f"  Rerouting Events: {metrics.rerouting_events[drone_id]}")
            console.print(f"  Average Velocity: {metrics.average_velocity[drone_id]:.2f}")

    except Exception as e:
        console.print(f"[bold red]Error during simulation: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()