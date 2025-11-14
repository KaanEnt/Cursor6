# RL-Based Support Ticket Assignment System

An intelligent reinforcement learning system for automatically assigning support tickets to agents based on their skills, workload, and ticket priority.

## Overview

This project implements a complete RL-based ticket assignment system using:
- **Custom Gymnasium Environment** for simulating ticket assignment scenarios
- **Multiple RL Algorithms** (PPO, A2C, DQN) via Stable-Baselines3 and Ray RLlib
- **Comprehensive Evaluation Metrics** for performance analysis
- **Interactive Streamlit Dashboard** for visualization and comparison

## Features

### Core Capabilities
- Synthetic ticket and agent data generation
- Custom RL environment with realistic reward functions
- Distributed training with Ray RLlib
- Vectorized training with Stable-Baselines3
- Real-time performance monitoring
- Baseline comparison (random vs RL policies)

### Key Metrics
- **Resolution Time**: Average time to resolve tickets
- **SLA Compliance**: Percentage of tickets meeting SLA thresholds
- **Workload Balance**: Distribution and fairness of agent assignments
- **Specialty Matching**: Rate of matching tickets to skilled agents

## Project Structure

```
.
├── data/
│   └── ticket_generator.py       # Synthetic data generation
├── environments/
│   └── ticket_env.py             # Custom Gymnasium environment
├── models/
│   ├── sb3_trainer.py            # Stable-Baselines3 training pipeline
│   ├── rllib_trainer.py          # Ray RLlib training pipeline
│   ├── saved_models/             # Trained model checkpoints
│   │   ├── sb3/
│   │   └── rllib/
│   └── logs/                     # Training logs
│       └── sb3/
├── utils/
│   ├── config.py                 # Configuration management
│   └── metrics.py                # Evaluation metrics
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── TESTING.md                    # Comprehensive testing guide
└── README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd Cursor6

# Activate your conda environment
conda activate VenvWeb

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Components

```bash
# Test data generator
python data/ticket_generator.py

# Test environment
python environments/ticket_env.py

# Test metrics
python utils/metrics.py
```

### 3. Train Models

**Stable-Baselines3 (Recommended for quick start):**
```bash
python models/sb3_trainer.py
```

**Ray RLlib (For distributed training):**
```bash
python models/rllib_trainer.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage Examples

### Custom Training

```python
from models.sb3_trainer import SB3Trainer
from utils.config import SB3Config, EnvironmentConfig

# Configure environment
env_config = EnvironmentConfig(
    num_agents=5,
    num_tickets=100,
    max_agent_load=10
)

# Configure training
sb3_config = SB3Config(
    algorithm="PPO",
    total_timesteps=100_000,
    n_envs=4,
    learning_rate=3e-4
)

# Train
trainer = SB3Trainer(env_config, sb3_config)
trainer.create_model()
trainer.train()
trainer.save_model()

# Evaluate
results = trainer.evaluate(n_episodes=10)
print(f"SLA Compliance: {results['sla_compliance']:.2%}")
```

### Generating Custom Data

```python
from data.ticket_generator import generate_data

# Generate tickets and agents
tickets_df, agents_df = generate_data(
    num_tickets=200,
    num_agents=8
)

# Inspect
print(tickets_df.head())
print(agents_df.head())
```

### Running Custom Evaluation

```python
from environments.ticket_env import TicketAssignmentEnv
from utils.metrics import TicketAssignmentMetrics

# Create environment
env = TicketAssignmentEnv(num_agents=5, num_tickets=100)

# Run episode with your policy
obs, info = env.reset()
done = False

while not done:
    action = your_policy(obs)  # Your custom policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Evaluate performance
metrics_calc = TicketAssignmentMetrics()
results = metrics_calc.evaluate_policy(
    env.tickets_df,
    env.agents_df,
    env.assignments
)

print(f"Resolution Time: {results['resolution_time']['mean']:.2f}")
print(f"SLA Compliance: {results['sla_compliance']['overall']:.2%}")
```

## Environment Details

### State Space
The observation includes:
- **Ticket features**: Type (one-hot), priority (one-hot), complexity, age
- **Agent states**: Current workload, average skill levels
- **Global features**: Queue length, time of day

### Action Space
Discrete action space where each action represents assigning the current ticket to a specific agent (0 to N-1).

### Reward Function
Rewards are calculated based on:
- **Skill match** (+): Assigning tickets to agents with high relevant skills
- **Priority urgency** (+): Quickly handling high-priority tickets
- **Workload balance** (-): Penalty for overloading agents
- **Complexity matching** (+): Matching complex tickets to skilled agents

### Episode Structure
Each episode simulates one day of ticket flow (configurable number of tickets, default 100).

## Configuration

All hyperparameters and settings can be customized via `utils/config.py`:

```python
from utils.config import ProjectConfig

# Load default config
config = ProjectConfig()

# Modify settings
config.environment.num_agents = 8
config.sb3.total_timesteps = 1_000_000
config.sb3.learning_rate = 1e-4

# Save custom config
config.save('my_config.yaml')

# Load config
config = ProjectConfig.load('my_config.yaml')
```

## Dashboard Features

### 📊 Overview
- System architecture and objectives
- Sample data visualization
- Ticket type and agent skill distribution

### 🎯 Training Monitor
- Real-time training curves
- Episode rewards and lengths
- Support for both SB3 and RLlib

### 🔬 Performance Comparison
- Side-by-side RL vs baseline comparison
- Comprehensive metrics visualization
- Performance improvement charts

### 🎬 Live Demo
- Interactive policy demonstration
- Real-time ticket assignment
- Workload distribution visualization

## Monitoring with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir=models/logs/sb3

# Open browser to http://localhost:6006
```

## Performance Tips

1. **Quick Testing**: Use 50K timesteps with 2-4 parallel environments
2. **Production Training**: Use 500K-1M timesteps with 4-8 environments
3. **System Resources**: Monitor RAM usage, especially with Ray
4. **GPU Usage**: Modify configs to enable GPU acceleration if available

## Troubleshooting

See [TESTING.md](TESTING.md) for detailed troubleshooting guide.

**Common issues:**
- Missing dependencies → `pip install -r requirements.txt`
- Ray initialization errors → Try `pip install "ray[rllib]" --force-reinstall`
- Slow training → Reduce environment count or timesteps
- Dashboard shows no data → Train a model first

## Next Steps

After initial setup:

1. Tune hyperparameters in configuration
2. Run longer training sessions (500K+ steps)
3. Compare different algorithms (PPO vs A2C vs DQN)
4. Customize reward function for your use case
5. Add domain-specific ticket types and attributes
6. Deploy trained model to production

## Technical Details

### Dependencies
- Python 3.9+
- Gymnasium (RL environment)
- Stable-Baselines3 (RL algorithms)
- Ray[rllib] (Distributed RL)
- Streamlit (Dashboard)
- Plotly (Visualizations)
- Pandas, NumPy (Data processing)

### Algorithms Supported
- **PPO** (Proximal Policy Optimization) - Recommended
- **A2C** (Advantage Actor-Critic)
- **DQN** (Deep Q-Network)
- **APPO** (Asynchronous PPO) - Ray only

### Training Features
- Vectorized environments for faster training
- Distributed workers with Ray
- Automatic checkpointing
- Evaluation callbacks
- TensorBoard integration

## Support

For detailed testing instructions, see [TESTING.md](TESTING.md)

---

**Built with Reinforcement Learning**
