"""
Configuration management for RL ticket assignment system.

This module centralizes all hyperparameters and configuration settings
for the training pipelines, environment, and evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml
import os


@dataclass
class EnvironmentConfig:
    """Configuration for the Ticket Assignment Environment."""
    num_agents: int = 5
    num_tickets: int = 100
    max_agent_load: int = 10
    render_mode: str = None


@dataclass
class SB3Config:
    """Configuration for Stable-Baselines3 training."""
    # Algorithm selection
    algorithm: str = "PPO"  # Options: PPO, A2C, DQN

    # Training parameters
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # For PPO/A2C
    batch_size: int = 64
    n_epochs: int = 10  # For PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95  # For PPO

    # Network architecture
    policy: str = "MlpPolicy"
    net_arch: list = None  # Will use default if None

    # Vectorized environments
    n_envs: int = 4

    # Logging and checkpointing
    log_interval: int = 10
    save_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    # Model save path
    model_dir: str = "models/saved_models/sb3"
    log_dir: str = "models/logs/sb3"

    def __post_init__(self):
        """Set default network architecture if not specified."""
        if self.net_arch is None:
            self.net_arch = [dict(pi=[256, 256], vf=[256, 256])]


@dataclass
class RLlibConfig:
    """Configuration for Ray RLlib training."""
    # Algorithm selection
    algorithm: str = "PPO"  # Options: PPO, APPO, DQN

    # Training parameters
    num_iterations: int = 500
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95

    # Environment settings
    num_workers: int = 4
    num_envs_per_worker: int = 1

    # Network architecture
    model_config: Dict[str, Any] = None

    # Checkpointing
    checkpoint_freq: int = 50
    checkpoint_dir: str = "models/saved_models/rllib"

    def __post_init__(self):
        """Set default model configuration if not specified."""
        if self.model_config is None:
            self.model_config = {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    # Baseline comparison
    run_baseline: bool = True
    baseline_episodes: int = 10

    # Metrics to track
    track_resolution_time: bool = True
    track_sla_compliance: bool = True
    track_agent_balance: bool = True
    track_specialty_match: bool = True

    # SLA thresholds (in time units)
    sla_threshold_low: int = 100
    sla_threshold_medium: int = 50
    sla_threshold_high: int = 20
    sla_threshold_critical: int = 10


@dataclass
class DashboardConfig:
    """Configuration for Streamlit dashboard."""
    # Page settings
    page_title: str = "RL Ticket Assignment System"
    page_icon: str = "🎫"
    layout: str = "wide"

    # Update intervals (in seconds)
    training_update_interval: int = 5

    # Visualization settings
    max_training_points: int = 1000
    plot_theme: str = "plotly_white"

    # Model loading
    default_sb3_model: str = "models/saved_models/sb3/best_model"
    default_rllib_checkpoint: str = "models/saved_models/rllib"


@dataclass
class ProjectConfig:
    """Main project configuration containing all sub-configs."""
    environment: EnvironmentConfig = None
    sb3: SB3Config = None
    rllib: RLlibConfig = None
    metrics: MetricsConfig = None
    dashboard: DashboardConfig = None

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.sb3 is None:
            self.sb3 = SB3Config()
        if self.rllib is None:
            self.rllib = RLlibConfig()
        if self.metrics is None:
            self.metrics = MetricsConfig()
        if self.dashboard is None:
            self.dashboard = DashboardConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': asdict(self.environment),
            'sb3': asdict(self.sb3),
            'rllib': asdict(self.rllib),
            'metrics': asdict(self.metrics),
            'dashboard': asdict(self.dashboard)
        }

    def save(self, filepath: str) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        filepath : str
            Path to save the configuration file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ProjectConfig':
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : str
            Path to the configuration file.

        Returns
        -------
        ProjectConfig
            Loaded configuration object.
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            sb3=SB3Config(**config_dict.get('sb3', {})),
            rllib=RLlibConfig(**config_dict.get('rllib', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            dashboard=DashboardConfig(**config_dict.get('dashboard', {}))
        )


def get_default_config() -> ProjectConfig:
    """
    Get the default project configuration.

    Returns
    -------
    ProjectConfig
        Default configuration object.
    """
    return ProjectConfig()


# Test configuration management
if __name__ == "__main__":
    print("=" * 80)
    print("Configuration Management - Test")
    print("=" * 80)
    print()

    # Create default configuration
    print("Creating default configuration...")
    config = get_default_config()
    print()

    # Display configuration
    print("Environment Configuration:")
    print(f"  Number of agents: {config.environment.num_agents}")
    print(f"  Number of tickets per episode: {config.environment.num_tickets}")
    print(f"  Max agent load: {config.environment.max_agent_load}")
    print()

    print("SB3 Training Configuration:")
    print(f"  Algorithm: {config.sb3.algorithm}")
    print(f"  Total timesteps: {config.sb3.total_timesteps:,}")
    print(f"  Learning rate: {config.sb3.learning_rate}")
    print(f"  Number of environments: {config.sb3.n_envs}")
    print()

    print("RLlib Training Configuration:")
    print(f"  Algorithm: {config.rllib.algorithm}")
    print(f"  Number of iterations: {config.rllib.num_iterations}")
    print(f"  Number of workers: {config.rllib.num_workers}")
    print()

    print("Metrics Configuration:")
    print(f"  Track SLA compliance: {config.metrics.track_sla_compliance}")
    print(f"  Run baseline: {config.metrics.run_baseline}")
    print()

    # Test saving configuration
    print("Testing configuration save/load...")
    save_path = "config_test.yaml"
    config.save(save_path)

    # Load configuration
    loaded_config = ProjectConfig.load(save_path)
    print(f"Configuration loaded successfully from {save_path}")
    print(f"Loaded SB3 algorithm: {loaded_config.sb3.algorithm}")
    print(f"Loaded environment num_agents: {loaded_config.environment.num_agents}")
    print()

    # Clean up test file
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Test file {save_path} removed")

    print()
    print("=" * 80)
    print("Configuration test completed successfully!")
    print("=" * 80)
