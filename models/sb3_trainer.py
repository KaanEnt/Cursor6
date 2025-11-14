"""
Stable-Baselines3 training pipeline for ticket assignment.

This module provides training functionality using Stable-Baselines3 algorithms
(PPO, A2C, DQN) for the ticket assignment environment.
"""

import os
from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from environments.ticket_env import TicketAssignmentEnv
from environments.ticket_env_temporal import TemporalTicketAssignmentEnv
from utils.config import SB3Config, EnvironmentConfig, get_default_config
from utils.metrics import TicketAssignmentMetrics


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at every step."""
        # Log episode info when available
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                # Log to tensorboard
                self.logger.record('rollout/ep_rew_mean_custom', np.mean(self.episode_rewards[-100:]))
                self.logger.record('rollout/ep_len_mean_custom', np.mean(self.episode_lengths[-100:]))

        return True


class SB3Trainer:
    """
    Trainer class for Stable-Baselines3 algorithms.

    This class handles training, evaluation, and model management for
    RL agents using the Stable-Baselines3 library.
    """

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        sb3_config: Optional[SB3Config] = None
    ):
        """
        Initialize the SB3 trainer.

        Parameters
        ----------
        env_config : EnvironmentConfig, optional
            Configuration for the environment.
        sb3_config : SB3Config, optional
            Configuration for SB3 training.
        """
        if env_config is None:
            default_config = get_default_config()
            env_config = default_config.environment

        if sb3_config is None:
            default_config = get_default_config()
            sb3_config = default_config.sb3

        self.env_config = env_config
        self.sb3_config = sb3_config

        # Create directories
        os.makedirs(sb3_config.model_dir, exist_ok=True)
        os.makedirs(sb3_config.log_dir, exist_ok=True)

        self.model = None
        self.env = None

    def _make_env(self):
        """Create a single environment instance."""
        def _init():
            if self.env_config.use_temporal:
                # Use fast temporal environment
                env = TemporalTicketAssignmentEnv(
                    num_agents=self.env_config.num_agents,
                    episode_steps=self.env_config.episode_steps,
                    tickets_per_step=self.env_config.tickets_per_step,
                    max_queue_size=self.env_config.max_queue_size
                )
            else:
                # Use simple instant assignment environment
                env = TicketAssignmentEnv(
                    num_agents=self.env_config.num_agents,
                    num_tickets=self.env_config.num_tickets,
                    max_agent_load=self.env_config.max_agent_load
                )
            env = Monitor(env)
            return env
        return _init

    def create_env(self, n_envs: Optional[int] = None) -> DummyVecEnv:
        """
        Create vectorized environments for training.

        Parameters
        ----------
        n_envs : int, optional
            Number of parallel environments. If None, uses config value.

        Returns
        -------
        DummyVecEnv or SubprocVecEnv
            Vectorized environment.
        """
        if n_envs is None:
            n_envs = self.sb3_config.n_envs

        if n_envs == 1:
            self.env = DummyVecEnv([self._make_env()])
        else:
            # Use SubprocVecEnv for parallel environments
            self.env = SubprocVecEnv([self._make_env() for _ in range(n_envs)])

        return self.env

    def create_model(self, algorithm: Optional[str] = None):
        """
        Create an RL model based on the specified algorithm.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm to use ('PPO', 'A2C', 'DQN'). If None, uses config value.

        Returns
        -------
        BaseAlgorithm
            The created model.
        """
        if algorithm is None:
            algorithm = self.sb3_config.algorithm

        if self.env is None:
            self.create_env()

        # Common kwargs for all algorithms
        common_kwargs = {
            'policy': self.sb3_config.policy,
            'env': self.env,
            'learning_rate': self.sb3_config.learning_rate,
            'gamma': self.sb3_config.gamma,
            'verbose': 1,
            'tensorboard_log': self.sb3_config.log_dir
        }

        # Algorithm-specific kwargs
        if algorithm == 'PPO':
            self.model = PPO(
                **common_kwargs,
                n_steps=self.sb3_config.n_steps,
                batch_size=self.sb3_config.batch_size,
                n_epochs=self.sb3_config.n_epochs,
                gae_lambda=self.sb3_config.gae_lambda,
                policy_kwargs={'net_arch': self.sb3_config.net_arch} if self.sb3_config.net_arch else None
            )
        elif algorithm == 'A2C':
            self.model = A2C(
                **common_kwargs,
                n_steps=self.sb3_config.n_steps,
                gae_lambda=self.sb3_config.gae_lambda,
                policy_kwargs={'net_arch': self.sb3_config.net_arch} if self.sb3_config.net_arch else None
            )
        elif algorithm == 'DQN':
            self.model = DQN(
                **common_kwargs,
                batch_size=self.sb3_config.batch_size,
                buffer_size=100000,
                learning_starts=1000,
                target_update_interval=1000
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from ['PPO', 'A2C', 'DQN']")

        print(f"Created {algorithm} model with {self.sb3_config.n_envs} parallel environments")
        return self.model

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback_list: Optional[list] = None
    ):
        """
        Train the model.

        Parameters
        ----------
        total_timesteps : int, optional
            Total number of timesteps to train for. If None, uses config value.
        callback_list : list, optional
            List of additional callbacks to use during training.

        Returns
        -------
        BaseAlgorithm
            The trained model.
        """
        if self.model is None:
            self.create_model()

        if total_timesteps is None:
            total_timesteps = self.sb3_config.total_timesteps

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.sb3_config.save_freq // self.sb3_config.n_envs,  # Adjust for vectorized env
            save_path=self.sb3_config.model_dir,
            name_prefix=f"{self.sb3_config.algorithm.lower()}_model"
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_env = DummyVecEnv([self._make_env()])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.sb3_config.model_dir,
            log_path=self.sb3_config.log_dir,
            eval_freq=self.sb3_config.eval_freq // self.sb3_config.n_envs,  # Adjust for vectorized env
            n_eval_episodes=self.sb3_config.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

        # Custom tensorboard callback
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)

        # Add any additional callbacks
        if callback_list:
            callbacks.extend(callback_list)

        # Combine all callbacks
        callback = CallbackList(callbacks)

        # Train the model
        print(f"Starting training for {total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=self.sb3_config.log_interval,
            progress_bar=True
        )

        print("Training completed!")
        return self.model

    def save_model(self, path: Optional[str] = None):
        """
        Save the trained model.

        Parameters
        ----------
        path : str, optional
            Path to save the model. If None, uses default path from config.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if path is None:
            path = os.path.join(self.sb3_config.model_dir, "final_model")

        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, algorithm: Optional[str] = None):
        """
        Load a trained model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        algorithm : str, optional
            Algorithm type ('PPO', 'A2C', 'DQN'). If None, uses config value.

        Returns
        -------
        BaseAlgorithm
            The loaded model.
        """
        if algorithm is None:
            algorithm = self.sb3_config.algorithm

        if algorithm == 'PPO':
            self.model = PPO.load(path)
        elif algorithm == 'A2C':
            self.model = A2C.load(path)
        elif algorithm == 'DQN':
            self.model = DQN.load(path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        print(f"Model loaded from {path}")
        return self.model

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Parameters
        ----------
        n_episodes : int, default=10
            Number of episodes to evaluate.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Train or load a model first.")

        # Create evaluation environment
        if self.env_config.use_temporal:
            eval_env = TemporalTicketAssignmentEnv(
                num_agents=self.env_config.num_agents,
                episode_steps=self.env_config.episode_steps,
                tickets_per_step=self.env_config.tickets_per_step,
                max_queue_size=self.env_config.max_queue_size
            )
        else:
            eval_env = TicketAssignmentEnv(
                num_agents=self.env_config.num_agents,
                num_tickets=self.env_config.num_tickets,
                max_agent_load=self.env_config.max_agent_load
            )

        metrics_calc = TicketAssignmentMetrics()
        results = []

        for episode in range(n_episodes):
            obs, info = eval_env.reset(seed=episode)
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                episode_reward += reward
                done = terminated or truncated

            # Calculate metrics for this episode
            episode_metrics = metrics_calc.evaluate_policy(
                eval_env.tickets_df,
                eval_env.agents_df,
                eval_env.assignments
            )
            episode_metrics['total_reward'] = episode_reward

            results.append(episode_metrics)

        # Aggregate results
        aggregated = {
            'resolution_time_mean': np.mean([r['resolution_time']['mean'] for r in results]),
            'sla_compliance': np.mean([r['sla_compliance']['overall'] for r in results]),
            'workload_std': np.mean([r['workload_balance']['std'] for r in results]),
            'specialty_match_rate': np.mean([r['specialty_match']['overall_match_rate'] for r in results]),
            'total_reward_mean': np.mean([r['total_reward'] for r in results]),
            'episodes': results
        }

        return aggregated


# Training script
if __name__ == "__main__":
    print("=" * 80)
    print("Stable-Baselines3 Training Pipeline")
    print("=" * 80)
    print()

    # Create trainer with default configuration
    trainer = SB3Trainer()

    print("Configuration:")
    print(f"  Algorithm: {trainer.sb3_config.algorithm}")
    print(f"  Environment: {trainer.env_config.num_agents} agents, {trainer.env_config.num_tickets} tickets")
    print(f"  Training timesteps: {trainer.sb3_config.total_timesteps:,}")
    print(f"  Parallel environments: {trainer.sb3_config.n_envs}")
    print()

    # Create and train model
    print("Creating model and starting training...")
    trainer.create_model()
    trainer.train()

    # Save the final model
    print("\nSaving model...")
    trainer.save_model()

    # Evaluate the trained model
    print("\nEvaluating trained model...")
    eval_results = trainer.evaluate(n_episodes=10)

    print("\nEvaluation Results:")
    print(f"  Average Resolution Time: {eval_results['resolution_time_mean']:.2f}")
    print(f"  SLA Compliance: {eval_results['sla_compliance']:.2%}")
    print(f"  Workload Balance (Std): {eval_results['workload_std']:.2f}")
    print(f"  Specialty Match Rate: {eval_results['specialty_match_rate']:.2%}")
    print(f"  Average Episode Reward: {eval_results['total_reward_mean']:.2f}")

    print()
    print("=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)
