"""
Custom Gymnasium Environment for Support Ticket Assignment.

This module implements a reinforcement learning environment that simulates
a SaaS support ticket assignment system where an RL agent learns to optimally
assign incoming tickets to available support agents.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ticket_generator import generate_data, TICKET_TYPES, PRIORITIES


class TicketAssignmentEnv(gym.Env):
    """
    A Gymnasium environment for learning optimal support ticket assignment.

    State Space
    -----------
    The observation is a flattened vector containing:
    - Current ticket features (type one-hot, priority one-hot, complexity, age)
    - Agent workloads (current number of assigned tickets per agent)
    - Agent average skills (average skill level per agent)
    - Queue length
    - Time of day (normalized hour)

    Action Space
    ------------
    Discrete space of size num_agents, where each action represents assigning
    the current ticket to a specific agent.

    Reward Function
    ---------------
    The reward is calculated based on:
    - Skill match: Positive reward for assigning to agents with high skill in ticket type
    - Workload balance: Penalty for overloading agents
    - Priority urgency: Higher rewards for quickly assigning high-priority tickets
    - Queue management: Penalty for long queue times

    Episode
    -------
    Each episode simulates one day of ticket flow (configurable number of tickets).
    The episode ends when all tickets in the queue have been assigned.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_agents: int = 5,
        num_tickets: int = 100,
        max_agent_load: int = 10,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Ticket Assignment Environment.

        Parameters
        ----------
        num_agents : int, default=5
            Number of support agents available for ticket assignment.
        num_tickets : int, default=100
            Number of tickets to process in one episode (simulates one day).
        max_agent_load : int, default=10
            Maximum number of tickets an agent should handle simultaneously.
        render_mode : str, optional
            Mode for rendering ('human' or None).
        """
        super().__init__()

        self.num_agents = num_agents
        self.num_tickets = num_tickets
        self.max_agent_load = max_agent_load
        self.render_mode = render_mode

        # Generate tickets and agents for the episode
        self.tickets_df, self.agents_df = generate_data(num_tickets, num_agents)

        # Episode state
        self.current_ticket_idx = 0
        self.agent_workloads = np.zeros(num_agents, dtype=np.int32)
        self.ticket_ages = np.zeros(num_tickets, dtype=np.float32)
        self.assignments = []  # Track (ticket_id, agent_id, reward) tuples
        self.current_step = 0

        # Time simulation (0-23 hours, normalized)
        self.current_hour = 9  # Start at 9 AM

        # Define action space: choose which agent to assign ticket to
        self.action_space = spaces.Discrete(num_agents)

        # Define observation space
        # Ticket features: type (5), priority (4), complexity (1), age (1) = 11
        # Agent features: workloads (num_agents), avg_skills (num_agents) = 2 * num_agents
        # Global features: queue_length (1), time_of_day (1) = 2
        # Total: 11 + 2*num_agents + 2
        obs_dim = 11 + 2 * num_agents + 2

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Priority weights for reward calculation
        self.priority_weights = {
            'Low': 1.0,
            'Medium': 2.0,
            'High': 3.0,
            'Critical': 5.0
        }

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation state.

        Returns
        -------
        np.ndarray
            Flattened observation vector.
        """
        if self.current_ticket_idx >= len(self.tickets_df):
            # Return zero observation if no tickets left
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        ticket = self.tickets_df.iloc[self.current_ticket_idx]

        # Ticket type one-hot encoding
        type_onehot = np.zeros(len(TICKET_TYPES), dtype=np.float32)
        type_idx = TICKET_TYPES.index(ticket['type'])
        type_onehot[type_idx] = 1.0

        # Priority one-hot encoding
        priority_onehot = np.zeros(len(PRIORITIES), dtype=np.float32)
        priority_idx = PRIORITIES.index(ticket['priority'])
        priority_onehot[priority_idx] = 1.0

        # Normalized complexity (1-5 -> 0.2-1.0)
        complexity_norm = ticket['complexity'] / 5.0

        # Normalized age (assuming max age is 1 hour per ticket in queue)
        age_norm = min(self.ticket_ages[self.current_ticket_idx] / self.num_tickets, 1.0)

        # Agent workloads normalized by max_agent_load
        workloads_norm = self.agent_workloads / self.max_agent_load

        # Agent average skills (average across all ticket types)
        agent_avg_skills = np.array([
            np.mean(list(self.agents_df.iloc[i]['skills'].values()))
            for i in range(self.num_agents)
        ], dtype=np.float32)

        # Queue length normalized
        remaining_tickets = len(self.tickets_df) - self.current_ticket_idx
        queue_length_norm = remaining_tickets / self.num_tickets

        # Time of day normalized (0-23 -> 0-1)
        time_norm = self.current_hour / 24.0

        # Concatenate all features
        obs = np.concatenate([
            type_onehot,
            priority_onehot,
            [complexity_norm],
            [age_norm],
            workloads_norm,
            agent_avg_skills,
            [queue_length_norm],
            [time_norm]
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional environment information.

        Returns
        -------
        dict
            Dictionary containing episode statistics and metadata.
        """
        return {
            'current_ticket_idx': self.current_ticket_idx,
            'tickets_processed': len(self.assignments),
            'agent_workloads': self.agent_workloads.copy(),
            'current_hour': self.current_hour,
            'avg_queue_age': np.mean(self.ticket_ages[:self.current_ticket_idx + 1]) if self.current_ticket_idx > 0 else 0
        }

    def _calculate_reward(self, ticket_idx: int, agent_idx: int) -> float:
        """
        Calculate the reward for assigning a ticket to an agent.

        Parameters
        ----------
        ticket_idx : int
            Index of the ticket being assigned.
        agent_idx : int
            Index of the agent receiving the assignment.

        Returns
        -------
        float
            The calculated reward value.
        """
        ticket = self.tickets_df.iloc[ticket_idx]
        agent = self.agents_df.iloc[agent_idx]

        # 1. Skill match reward: How well does agent's skill match ticket type?
        ticket_type = ticket['type']
        agent_skill = agent['skills'][ticket_type]
        skill_reward = agent_skill * 10.0  # Scale to 5-10 range

        # 2. Priority urgency reward: Higher priority tickets get more reward for quick assignment
        priority = ticket['priority']
        priority_weight = self.priority_weights[priority]

        # Age penalty: Older tickets in queue should be prioritized
        age_penalty = self.ticket_ages[ticket_idx] * 0.1

        urgency_reward = priority_weight * (1.0 + age_penalty)

        # 3. Workload balance penalty: Penalize overloading agents
        workload_ratio = self.agent_workloads[agent_idx] / self.max_agent_load
        if workload_ratio > 0.8:
            workload_penalty = -5.0 * (workload_ratio - 0.8)
        else:
            workload_penalty = 0.5 * (1.0 - workload_ratio)  # Small bonus for using less loaded agents

        # 4. Complexity consideration: More complex tickets to more skilled agents
        complexity = ticket['complexity']
        complexity_match = agent_skill * complexity
        complexity_reward = complexity_match * 0.5

        # Total reward
        total_reward = skill_reward + urgency_reward + workload_penalty + complexity_reward

        return float(total_reward)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options for reset.

        Returns
        -------
        tuple
            (observation, info) - Initial observation and info dict.
        """
        super().reset(seed=seed)

        # Generate new tickets and agents
        self.tickets_df, self.agents_df = generate_data(self.num_tickets, self.num_agents)

        # Reset episode state
        self.current_ticket_idx = 0
        self.agent_workloads = np.zeros(self.num_agents, dtype=np.int32)
        self.ticket_ages = np.zeros(self.num_tickets, dtype=np.float32)
        self.assignments = []
        self.current_step = 0
        self.current_hour = 9  # Start at 9 AM

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int
            The agent index to assign the current ticket to.

        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if action < 0 or action >= self.num_agents:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_agents - 1}")

        # Calculate reward for this assignment
        reward = self._calculate_reward(self.current_ticket_idx, action)

        # Record assignment
        ticket_id = self.tickets_df.iloc[self.current_ticket_idx]['ticket_id']
        self.assignments.append((ticket_id, action, reward))

        # Update agent workload
        self.agent_workloads[action] += 1

        # Move to next ticket
        self.current_ticket_idx += 1
        self.current_step += 1

        # Update ticket ages (all remaining tickets age by 1 unit)
        if self.current_ticket_idx < len(self.tickets_df):
            self.ticket_ages[self.current_ticket_idx:] += 1

        # Advance time (simulate ~5 minutes per ticket)
        self.current_hour = (9 + (self.current_step * 5 / 60)) % 24

        # Check if episode is done
        terminated = self.current_ticket_idx >= len(self.tickets_df)
        truncated = False

        # Get next observation
        observation = self._get_obs()
        info = self._get_info()

        # Add final statistics to info if episode is done
        if terminated:
            info['episode_reward'] = sum([r for _, _, r in self.assignments])
            info['avg_reward'] = info['episode_reward'] / len(self.assignments)
            info['workload_std'] = np.std(self.agent_workloads)

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment state.

        For human mode, prints current episode statistics.
        """
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step} | Hour: {self.current_hour:.1f}")
            print(f"Tickets Processed: {self.current_ticket_idx}/{self.num_tickets}")
            print(f"Agent Workloads: {self.agent_workloads}")
            if self.assignments:
                avg_reward = np.mean([r for _, _, r in self.assignments])
                print(f"Average Reward: {avg_reward:.2f}")
            print(f"{'='*60}")

    def close(self):
        """Clean up environment resources."""
        pass


# Test the environment
if __name__ == "__main__":
    print("=" * 80)
    print("Ticket Assignment Environment - Test")
    print("=" * 80)
    print()

    # Create environment
    env = TicketAssignmentEnv(num_agents=5, num_tickets=20, render_mode="human")

    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print()

    # Test reset
    print("Testing environment reset...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    print()

    # Run a few random steps
    print("Running 10 random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}")

        if terminated:
            print("Episode finished early!")
            break

    print()
    print(f"Total reward for 10 steps: {total_reward:.2f}")
    print()

    # Test full episode with random policy
    print("Testing full episode with random policy...")
    obs, info = env.reset(seed=123)
    episode_reward = 0
    step_count = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    env.render()
    print(f"\nEpisode Statistics:")
    print(f"  Total Steps: {step_count}")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Average Reward: {episode_reward/step_count:.2f}")
    print(f"  Final Agent Workloads: {info['agent_workloads']}")
    print(f"  Workload Std Dev: {info['workload_std']:.2f}")
    print()

    # Test with check_env from stable_baselines3 if available
    try:
        from stable_baselines3.common.env_checker import check_env
        print("Running Stable-Baselines3 environment checker...")
        check_env(env, warn=True)
        print("Environment passed all checks!")
    except ImportError:
        print("Stable-Baselines3 not installed. Skipping env checker.")
    except Exception as e:
        print(f"Environment check failed: {e}")

    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
