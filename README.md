# 🚁 Probability Shadow Pathfinding System (PSPS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A simulation framework for autonomous drone navigation using probability shadow pathfinding. This project implements advanced uncertainty-aware path planning with real-time collision risk assessment and dynamic rerouting.

![Simulation Preview](simulation.gif)

## 🌟 Key Features

### Probability Shadow Navigation
- **Dynamic Risk Assessment**: Real-time collision probability calculation
- **Uncertainty Propagation**: Growing uncertainty shadows for future positions
- **Adaptive Rerouting**: Intelligent path adjustment based on risk assessment

### Advanced Analytics
- **Real-time Metrics**: Comprehensive tracking of system performance
- **Collision Risk Analysis**: Detailed probability-based risk assessment
- **Mission Progress Tracking**: Real-time completion monitoring
- **Path Optimization**: Analysis of route efficiency and safety

### Visualization
- **3D Shadow Visualization**: Real-time display of probability shadows
- **Risk Mapping**: Visual representation of collision risks
- **Path History**: Track historical movements and decisions
- **Analytics Dashboard**: Real-time performance metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/probability-shadow-pathfinding.git
cd probability-shadow-pathfinding

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Basic Usage

```python
from psps.simulation import ProbabilityShadowSimulation

# Create and run a simulation
sim = ProbabilityShadowSimulation(space_size=(50, 50, 30))
sim.run()
```

## 🎮 Advanced Usage

### Custom Configuration

```python
from psps.simulation import ProbabilityShadowGrid
import numpy as np

# Initialize the grid
grid = ProbabilityShadowGrid(space_size=(50, 50, 30))

# Configure simulation parameters
grid.prediction_steps = 20
grid.uncertainty_growth_rate = 0.08
grid.collision_threshold = 0.25
grid.safe_distance = 4.0
grid.time_horizon = 6.0

# Add drones with specific configurations
grid.initialize_drone(
    drone_id="drone_1",
    start=np.array([10, 10, 5]),
    goal=np.array([40, 40, 25])
)
```

### Analytics Configuration

```python
# Configure analytics collection
simulation.enable_analytics({
    'collision_risk': True,
    'path_efficiency': True,
    'mission_progress': True,
    'rerouting_events': True
})
```

## 🔧 Technical Details

### Core Components

1. **Probability Shadow Generation**
   - Multivariate normal distribution modeling
   - Uncertainty growth prediction
   - Dynamic shadow point generation

2. **Collision Risk Assessment**
   - Probabilistic collision detection
   - Risk threshold management
   - Real-time path validation

3. **Path Planning**
   - Risk-aware route planning
   - Dynamic obstacle avoidance
   - Efficient alternative path finding

### Performance Optimization

- Vectorized operations for shadow calculations
- Efficient spatial queries
- Optimized probability computations
- Streamlined visualization rendering

## 📊 Applications

- **Urban Air Mobility**: Safe autonomous drone navigation
- **Risk Assessment**: Probabilistic collision prediction
- **Path Planning**: Uncertainty-aware route optimization
- **Multi-Agent Systems**: Coordinated navigation with uncertainty
- **Safety Analysis**: Risk assessment for autonomous systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Probability theory concepts from advanced robotics research
- Visualization components built on Matplotlib and NumPy
- Statistical computations powered by SciPy
- Special thanks to all contributors and the open-source community

---