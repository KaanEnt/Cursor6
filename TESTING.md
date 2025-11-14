# Testing Guide

This document contains commands to run after each implementation phase.

## Quick Start Commands

```bash
# 1. Install dependencies
conda activate VenvWeb
pip install -r requirements.txt

# 2. Test temporal environment (RECOMMENDED)
python environments/ticket_env_temporal.py

# 3. Test config
python utils/config.py

# 4. Train model (5-10 minutes)
python models/sb3_trainer.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## Important: Fast Training Configuration

**This project is optimized for fast training (target: 10-20 minutes per run):**

- **Temporal Environment**: Realistic time-based simulation with 30-step episodes
- **Fast Training Defaults**:
  - 50K timesteps (vs 500K in typical RL)
  - 4 agents (vs 5)
  - 2 parallel environments (vs 4-8)
  - Coarse time granularity (10-15 min blocks)

**Two Environment Options:**
1. **Temporal (DEFAULT, RECOMMENDED)**: Realistic queue dynamics, agent availability, time-based resolution
2. **Simple**: Instant assignment, formula-based evaluation (for comparison)

Set `use_temporal=True` in config to use the temporal environment (default).

---

## Phase 1: Initial Setup & Data Generator

### Setup Commands

```bash
# Activate your conda environment
conda activate VenvWeb

# Install dependencies
pip install -r requirements.txt
```

### Testing the Data Generator

```bash
# Run the ticket generator test script
python data/ticket_generator.py
```

### Expected Output

The test script should display:
- 100 generated tickets with their attributes (ticket_id, type, priority, complexity)
- Distribution statistics for ticket types and priorities
- 5 agent profiles with skill levels for each ticket type (0.5 to 1.0 range)
- Edge case test results showing empty DataFrames work correctly

### Manual Testing (Optional)

You can also test the generator interactively in a Python shell:

```python
from data.ticket_generator import generate_data

# Generate custom dataset
tickets, agents = generate_data(num_tickets=50, num_agents=3)

# Inspect the data
print(tickets.head())
print(tickets.info())
print(agents.head())
print(agents['skills'][0])  # View first agent's skills
```

---

## Phase 2: Gymnasium Environment & Utilities

### Testing the Custom Gymnasium Environment

**Simple Environment (instant assignment):**
```bash
python environments/ticket_env.py
```

**Expected Output:**
- Action and observation space details
- 10 random steps with rewards
- Full episode statistics (workload distribution, total reward)
- Stable-Baselines3 environment checker validation (if SB3 is installed)

**Temporal Environment (realistic, time-based - RECOMMENDED):**
```bash
python environments/ticket_env_temporal.py
```

**Expected Output:**
- 30-step episode simulation
- Agent availability tracking (busy/free states)
- Queue management with wait times
- SLA compliance tracking
- Realistic resolution times over multiple steps
- Environment passes SB3 checker

**Key Differences:**
- Temporal: Agents have busy periods, tickets wait in queue, realistic time flow
- Simple: Instant assignment, formula-based resolution times

### Testing Configuration Management

```bash
# Run the config test script
python utils/config.py
```

### Expected Output

- Default configuration values for all components
- Successful save/load of configuration to YAML
- Verification that loaded config matches saved config

### Testing Evaluation Metrics

```bash
# Run the metrics test script (this will take a few minutes)
python utils/metrics.py
```

### Expected Output

- 5 episodes of random baseline policy evaluation
- Averaged metrics: resolution time, SLA compliance, workload balance, specialty match
- Detailed breakdown for one episode showing per-priority and per-ticket-type metrics

---

## Phase 3: Training Pipelines

### Prerequisites

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Testing Stable-Baselines3 Training

**Fast Training (DEFAULT - optimized for speed):**

```bash
# This will train for 50K timesteps (takes ~5-10 minutes)
# Uses temporal environment by default
python models/sb3_trainer.py
```

**Expected Output:**
- Model creation confirmation (PPO with 2 parallel envs)
- Training progress with episode rewards
- Checkpoint saves every 5K steps
- Final evaluation metrics after training
- Model saved to `models/saved_models/sb3/`
- **Training time: 5-10 minutes** on modern CPU

**Custom Training (Optional):**

```python
from models.sb3_trainer import SB3Trainer
from utils.config import SB3Config, EnvironmentConfig

# Customize environment
env_config = EnvironmentConfig(
    use_temporal=True,  # Use temporal environment (recommended)
    num_agents=4,
    episode_steps=30,  # 30 time steps per episode
    tickets_per_step=1.0
)

# Customize training
sb3_config = SB3Config(
    algorithm="PPO",
    total_timesteps=50_000,  # Fast training
    n_envs=2
)

# Train
trainer = SB3Trainer(env_config, sb3_config)
trainer.create_model()
trainer.train()
trainer.save_model()

# Evaluate
results = trainer.evaluate(n_episodes=5)
print(f"SLA Compliance: {results['sla_compliance']:.1%}")
```

### Testing Ray RLlib Training

**Fast Training (DEFAULT - optimized for speed):**

```bash
# This will train for 100 iterations (takes ~10-15 minutes)
# Uses temporal environment by default
python models/rllib_trainer.py
```

**Expected Output:**
- Ray initialization
- Algorithm creation with 2 workers
- Training progress every 10 iterations
- Checkpoint saves every 20 iterations
- Final evaluation metrics
- Checkpoints saved to `models/saved_models/rllib/`
- **Training time: 10-15 minutes** on modern CPU

**Note:** Ray may use significant system resources. The fast config uses 2 workers (vs 4) to reduce overhead.

---

## Phase 4: Streamlit Dashboard

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/app.py
```

### Expected Behavior

The dashboard should open in your browser with the following features:

**📊 Overview Tab:**
- System objective and approach description
- Sample ticket and agent data visualization
- Ticket type distribution pie chart
- Agent skill level bar charts

**🎯 Training Monitor Tab:**
- SB3 and RLlib tabs for training progress
- TensorBoard log integration (if logs available)
- Instructions for starting training if no data exists

**🔬 Performance Comparison Tab:**
- Button to run baseline evaluation
- Side-by-side metrics comparison (RL vs Random)
- Performance improvement bar chart
- Detailed results in expandable section

**🎬 Live Demo Tab:**
- Interactive policy demonstration
- Real-time ticket assignment visualization
- Live metrics display
- Workload distribution chart

### Testing Dashboard Features

1. **Navigate to Overview** - Verify data tables and charts load correctly
2. **Run Performance Comparison** - Click "Run Evaluation" and verify metrics appear
3. **Run Live Demo** - Select "Random Baseline" and click "Start Demo"
4. **Adjust Settings** - Use sidebar to change num_agents and num_tickets, verify updates

---

## Full Pipeline Test

### End-to-End Workflow

**1. Generate Data:**
```bash
python data/ticket_generator.py
```

**2. Test Environment:**
```bash
python environments/ticket_env.py
```

**3. Train SB3 Model (Quick):**
```python
from models.sb3_trainer import SB3Trainer
from utils.config import SB3Config, EnvironmentConfig

env_config = EnvironmentConfig(num_agents=5, num_tickets=50)
sb3_config = SB3Config(total_timesteps=50_000, n_envs=2)

trainer = SB3Trainer(env_config, sb3_config)
trainer.create_model()
trainer.train()
trainer.save_model("models/saved_models/sb3/quick_test_model")

# Evaluate
results = trainer.evaluate(n_episodes=5)
print(f"Average Reward: {results['total_reward_mean']:.2f}")
print(f"SLA Compliance: {results['sla_compliance']:.1%}")
```

**4. Launch Dashboard:**
```bash
streamlit run dashboard/app.py
```

**5. Verify in Dashboard:**
- Check if training logs appear in Training Monitor
- Run performance comparison
- Test live demo

---

## Monitoring Training with TensorBoard

If you want to monitor training in real-time:

```bash
# In a separate terminal, run TensorBoard
tensorboard --logdir=models/logs/sb3

# Open browser to http://localhost:6006
```

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'gymnasium'`
**Solution:**
```bash
pip install -r requirements.txt
```

**Issue:** Ray fails to initialize
**Solution:**
```bash
# Ray may need to be reinstalled
pip uninstall ray
pip install "ray[rllib]"
```

**Issue:** Streamlit shows "No training data available"
**Solution:** Train a model first using the training pipeline commands above

**Issue:** Training is very slow
**Solution:**
- Reduce `num_tickets` in environment config
- Reduce `n_envs` or `num_workers` in training config
- Use smaller `total_timesteps` for testing

### Performance Tips

1. **For quick testing:** Use 50K timesteps and 2-4 environments
2. **For serious training:** Use 500K+ timesteps and 4-8 environments
3. **Monitor system resources:** Ray can be resource-intensive
4. **Use GPU:** If available, modify training configs to use GPU

---

## Next Steps

After verifying all components work:

1. **Tune hyperparameters** in `utils/config.py`
2. **Run longer training** (500K-1M timesteps)
3. **Compare algorithms** (PPO vs A2C vs DQN)
4. **Customize reward function** in `environments/ticket_env.py`
5. **Add more ticket types** or agent attributes
6. **Deploy trained model** to production environment
