"""
Ray RLlib training pipeline for ticket assignment.

This module provides distributed training functionality using Ray RLlib
algorithms for the ticket assignment environment.
"""

import os
from typing import Optional, Dict, Any
import numpy as np

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.env_context import EnvContext

from environments.ticket_env import TicketAssignmentEnv
from environments.ticket_env_temporal import TemporalTicketAssignmentEnv
from utils.config import RLlibConfig, EnvironmentConfig, get_default_config
from utils.metrics import TicketAssignmentMetrics


def env_creator(env_config: EnvContext):
    """
    Environment creator function for Ray RLlib.

    Parameters
    ----------
    env_config : EnvContext
        Environment configuration from Ray.

    Returns
    -------
    TicketAssignmentEnv or TemporalTicketAssignmentEnv
        Created environment instance.
    """
    use_temporal = env_config.get('use_temporal', True)

    if use_temporal:
        return TemporalTicketAssignmentEnv(
            num_agents=env_config.get('num_agents', 4),
            episode_steps=env_config.get('episode_steps', 30),
            tickets_per_step=env_config.get('tickets_per_step', 1.0),
            max_queue_size=env_config.get('max_queue_size', 20)
        )
    else:
        return TicketAssignmentEnv(
            num_agents=env_config.get('num_agents', 4),
            num_tickets=env_config.get('num_tickets', 100),
            max_agent_load=env_config.get('max_agent_load', 10)
        )


class RLlibTrainer:
    """
    Trainer class for Ray RLlib algorithms.

    This class handles distributed training, evaluation, and model management
    for RL agents using the Ray RLlib library.
    """

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        rllib_config: Optional[RLlibConfig] = None
    ):
        """
        Initialize the RLlib trainer.

        Parameters
        ----------
        env_config : EnvironmentConfig, optional
            Configuration for the environment.
        rllib_config : RLlibConfig, optional
            Configuration for RLlib training.
        """
        if env_config is None:
            default_config = get_default_config()
            env_config = default_config.environment

        if rllib_config is None:
            default_config = get_default_config()
            rllib_config = default_config.rllib

        self.env_config = env_config
        self.rllib_config = rllib_config

        # Create directories
        os.makedirs(rllib_config.checkpoint_dir, exist_ok=True)

        self.algorithm = None
        self.env_name = "TicketAssignmentEnv"

        # Initialize Ray (if not already initialized)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Register environment
        from ray.tune.registry import register_env
        register_env(self.env_name, env_creator)

    def create_config(self, algorithm: Optional[str] = None):
        """
        Create RLlib algorithm configuration.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm to use ('PPO', 'APPO', 'DQN'). If None, uses config value.

        Returns
        -------
        AlgorithmConfig
            The created algorithm configuration.
        """
        if algorithm is None:
            algorithm = self.rllib_config.algorithm

        # Environment config for RLlib
        env_config = {
            'use_temporal': self.env_config.use_temporal,
            'num_agents': self.env_config.num_agents,
            # Simple environment settings
            'num_tickets': self.env_config.num_tickets,
            'max_agent_load': self.env_config.max_agent_load,
            # Temporal environment settings
            'episode_steps': self.env_config.episode_steps,
            'tickets_per_step': self.env_config.tickets_per_step,
            'max_queue_size': self.env_config.max_queue_size
        }

        # Create base config based on algorithm
        if algorithm == 'PPO':
            config = (
                PPOConfig()
                .environment(self.env_name, env_config=env_config)
                .framework("torch")
                .rollouts(
                    num_rollout_workers=self.rllib_config.num_workers,
                    num_envs_per_worker=self.rllib_config.num_envs_per_worker
                )
                .training(
                    train_batch_size=self.rllib_config.train_batch_size,
                    sgd_minibatch_size=self.rllib_config.sgd_minibatch_size,
                    num_sgd_iter=self.rllib_config.num_sgd_iter,
                    lr=self.rllib_config.lr,
                    gamma=self.rllib_config.gamma,
                    lambda_=self.rllib_config.lambda_,
                    model=self.rllib_config.model_config
                )
                .resources(num_gpus=0)
                .evaluation(
                    evaluation_interval=10,
                    evaluation_num_workers=1,
                    evaluation_duration=5
                )
            )

        elif algorithm == 'APPO':
            config = (
                APPOConfig()
                .environment(self.env_name, env_config=env_config)
                .framework("torch")
                .rollouts(
                    num_rollout_workers=self.rllib_config.num_workers,
                    num_envs_per_worker=self.rllib_config.num_envs_per_worker
                )
                .training(
                    train_batch_size=self.rllib_config.train_batch_size,
                    minibatch_size=self.rllib_config.sgd_minibatch_size,
                    num_sgd_iter=self.rllib_config.num_sgd_iter,
                    lr=self.rllib_config.lr,
                    gamma=self.rllib_config.gamma,
                    lambda_=self.rllib_config.lambda_,
                    model=self.rllib_config.model_config
                )
                .resources(num_gpus=0)
            )

        elif algorithm == 'DQN':
            config = (
                DQNConfig()
                .environment(self.env_name, env_config=env_config)
                .framework("torch")
                .rollouts(
                    num_rollout_workers=self.rllib_config.num_workers
                )
                .training(
                    train_batch_size=self.rllib_config.sgd_minibatch_size,
                    lr=self.rllib_config.lr,
                    gamma=self.rllib_config.gamma,
                    model=self.rllib_config.model_config
                )
                .resources(num_gpus=0)
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from ['PPO', 'APPO', 'DQN']")

        return config

    def train(
        self,
        num_iterations: Optional[int] = None,
        checkpoint_freq: Optional[int] = None,
        algorithm: Optional[str] = None
    ):
        """
        Train the model using Ray RLlib.

        Parameters
        ----------
        num_iterations : int, optional
            Number of training iterations. If None, uses config value.
        checkpoint_freq : int, optional
            Frequency of checkpointing. If None, uses config value.
        algorithm : str, optional
            Algorithm to use. If None, uses config value.

        Returns
        -------
        Algorithm
            The trained algorithm instance.
        """
        if num_iterations is None:
            num_iterations = self.rllib_config.num_iterations

        if checkpoint_freq is None:
            checkpoint_freq = self.rllib_config.checkpoint_freq

        if algorithm is None:
            algorithm = self.rllib_config.algorithm

        # Create config
        config = self.create_config(algorithm)

        # Build algorithm
        self.algorithm = config.build()

        print(f"Starting {algorithm} training for {num_iterations} iterations...")
        print(f"Workers: {self.rllib_config.num_workers}")
        print(f"Checkpointing every {checkpoint_freq} iterations to {self.rllib_config.checkpoint_dir}")
        print()

        # Training loop
        for iteration in range(num_iterations):
            result = self.algorithm.train()

            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Episode Reward Mean: {result.get('episode_reward_mean', 0):.2f}")
                print(f"  Episode Length Mean: {result.get('episode_len_mean', 0):.2f}")
                print(f"  Timesteps Total: {result.get('timesteps_total', 0)}")
                print()

            # Checkpoint
            if (iteration + 1) % checkpoint_freq == 0:
                checkpoint_path = self.algorithm.save(self.rllib_config.checkpoint_dir)
                print(f"Checkpoint saved to: {checkpoint_path}")

        # Final checkpoint
        final_checkpoint = self.algorithm.save(self.rllib_config.checkpoint_dir)
        print(f"\nTraining completed! Final checkpoint: {final_checkpoint}")

        return self.algorithm

    def load_checkpoint(self, checkpoint_path: str, algorithm: Optional[str] = None):
        """
        Load a trained algorithm from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint directory.
        algorithm : str, optional
            Algorithm type. If None, uses config value.

        Returns
        -------
        Algorithm
            The loaded algorithm instance.
        """
        if algorithm is None:
            algorithm = self.rllib_config.algorithm

        config = self.create_config(algorithm)
        self.algorithm = config.build()
        self.algorithm.restore(checkpoint_path)

        print(f"Algorithm loaded from checkpoint: {checkpoint_path}")
        return self.algorithm

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained algorithm.

        Parameters
        ----------
        n_episodes : int, default=10
            Number of episodes to evaluate.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """
        if self.algorithm is None:
            raise ValueError("No algorithm to evaluate. Train or load an algorithm first.")

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
                action = self.algorithm.compute_single_action(obs)
                obs, reward, terminated, truncated, info = eval_env.step(action)
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

    def shutdown(self):
        """Shutdown Ray and cleanup resources."""
        if self.algorithm:
            self.algorithm.stop()
        ray.shutdown()


# Training script
if __name__ == "__main__":
    print("=" * 80)
    print("Ray RLlib Training Pipeline")
    print("=" * 80)
    print()

    # Create trainer with default configuration
    trainer = RLlibTrainer()

    print("Configuration:")
    print(f"  Algorithm: {trainer.rllib_config.algorithm}")
    print(f"  Environment: {trainer.env_config.num_agents} agents, {trainer.env_config.num_tickets} tickets")
    print(f"  Training iterations: {trainer.rllib_config.num_iterations}")
    print(f"  Workers: {trainer.rllib_config.num_workers}")
    print()

    # Train model
    print("Starting training...")
    try:
        trainer.train()

        # Evaluate the trained model
        print("\nEvaluating trained model...")
        eval_results = trainer.evaluate(n_episodes=10)

        print("\nEvaluation Results:")
        print(f"  Average Resolution Time: {eval_results['resolution_time_mean']:.2f}")
        print(f"  SLA Compliance: {eval_results['sla_compliance']:.2%}")
        print(f"  Workload Balance (Std): {eval_results['workload_std']:.2f}")
        print(f"  Specialty Match Rate: {eval_results['specialty_match_rate']:.2%}")
        print(f"  Average Episode Reward: {eval_results['total_reward_mean']:.2f}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nShutting down Ray...")
        trainer.shutdown()

    print()
    print("=" * 80)
    print("Training pipeline completed!")
    print("=" * 80)
