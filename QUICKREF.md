# Quick Reference Card

## ⚡ 30-Second Start

```bash
conda activate VenvWeb
pip install gymnasium stable-baselines3 pandas numpy
python demo_simple.py
```

---

## 🎯 Common Commands

```bash
# Demo & Testing
python demo_simple.py                        # 2-min simple demo
python environments/ticket_env_temporal.py   # Test temporal env
python utils/config.py                       # Test config system

# Training
python models/sb3_trainer.py                 # Train SB3 (5-10 min)
python models/rllib_trainer.py               # Train RLlib (10-15 min)

# Dashboard
streamlit run dashboard/app.py               # Launch dashboard

# Monitoring
tensorboard --logdir=models/logs/sb3         # View training curves
```

---

## 📊 Three Environments

| Environment | File | Use Case | Speed |
|-------------|------|----------|-------|
| **Simple** | `ticket_env.py` | Learning, demos | ⚡⚡⚡ |
| **Temporal** | `ticket_env_temporal.py` | Production, research | ⚡⚡ |
| **Demo** | `demo_simple.py` | Quick proof | ⚡⚡⚡ |

---

## 🔧 Configuration Quick Edit

```python
from utils.config import ProjectConfig

# Get defaults
config = ProjectConfig()

# Customize
config.environment.use_temporal = True
config.environment.num_agents = 4
config.environment.episode_steps = 30

config.sb3.total_timesteps = 50_000
config.sb3.n_envs = 2

# Save
config.save('my_config.yaml')

# Load
config = ProjectConfig.load('my_config.yaml')
```

---

## 🎓 Training Snippets

### Quick SB3 Training
```python
from models.sb3_trainer import SB3Trainer
from utils.config import SB3Config, EnvironmentConfig

env_config = EnvironmentConfig(use_temporal=True, num_agents=4)
sb3_config = SB3Config(total_timesteps=50_000, n_envs=2)

trainer = SB3Trainer(env_config, sb3_config)
trainer.create_model()
trainer.train()
trainer.save_model()

results = trainer.evaluate(n_episodes=5)
print(f"SLA Compliance: {results['sla_compliance']:.1%}")
```

### Quick Environment Test
```python
from environments.ticket_env_temporal import TemporalTicketAssignmentEnv

env = TemporalTicketAssignmentEnv(num_agents=4, episode_steps=30)
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

print(f"Queue: {info['queue_length']}, Completed: {info['tickets_completed']}")
```

---

## 📈 Default Training Times

| Task | Time | Output |
|------|------|--------|
| Demo script | 2 min | Baseline + trained comparison |
| SB3 training | 5-10 min | 50K steps, ~1,600 episodes |
| RLlib training | 10-15 min | 100 iterations, checkpoints |
| Dashboard launch | 5 sec | Interactive visualizations |

---

## 🎨 Dashboard Tabs

| Tab | What You See |
|-----|--------------|
| **📊 Overview** | System description, sample data, distributions |
| **🎯 Training Monitor** | Real-time training curves (SB3/RLlib) |
| **🔬 Performance Comparison** | RL vs Random baseline metrics |
| **🎬 Live Demo** | Interactive ticket assignment simulation |

---

## 🐛 Quick Troubleshooting

```bash
# Import errors
pip install -r requirements.txt

# Slow training
# Reduce in utils/config.py:
# - total_timesteps: 50K → 10K
# - n_envs: 2 → 1
# - episode_steps: 30 → 20

# Ray issues
pip uninstall ray && pip install "ray[rllib]"

# Gymnasium issues
pip install --upgrade gymnasium
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `demo_simple.py` | 2-min demo (start here!) |
| `environments/ticket_env_temporal.py` | Main temporal environment |
| `models/sb3_trainer.py` | SB3 training pipeline |
| `utils/config.py` | All configuration settings |
| `dashboard/app.py` | Streamlit visualization |
| `TESTING.md` | Comprehensive testing guide |
| `ARCHITECTURE.md` | Design decisions & comparisons |

---

## 🚀 Optimization Tips

### For Faster Training
```python
env_config.episode_steps = 20  # Shorter episodes
sb3_config.total_timesteps = 20_000  # Fewer steps
sb3_config.n_envs = 1  # No parallelization
```

### For Better Performance
```python
env_config.episode_steps = 50  # Longer episodes
sb3_config.total_timesteps = 200_000  # More steps
sb3_config.n_envs = 4  # More parallelization
sb3_config.learning_rate = 1e-4  # Lower LR
```

### For Debugging
```python
env_config.render_mode = "human"  # Visual feedback
env = TicketAssignmentEnv(..., render_mode="human")
```

---

## 📊 Expected Results

### Random Policy (Baseline)
- Avg Reward: ~5-8
- SLA Compliance: 60-70%
- Workload Balance: σ = 2-3
- Skill Match: 65-70%

### RL Policy (50K steps)
- Avg Reward: ~12-15 (+80-100%)
- SLA Compliance: 85-95%
- Workload Balance: σ = 1-2
- Skill Match: 85-95%

---

## 🔗 Quick Links

- **Getting Started**: [README.md](README.md)
- **Testing Guide**: [TESTING.md](TESTING.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Original References**: `.specstory/reference/`

---

## 💡 Pro Tips

1. **Start with the demo**: `python demo_simple.py` gives you instant gratification
2. **Use temporal for real work**: Much more realistic evaluation
3. **Monitor with TensorBoard**: See training progress in real-time
4. **Save checkpoints**: They're saved every 5K steps automatically
5. **Compare algorithms**: Try PPO, A2C, DQN (PPO usually wins)

---

**Need more help?** Check [TESTING.md](TESTING.md) for detailed guides!
