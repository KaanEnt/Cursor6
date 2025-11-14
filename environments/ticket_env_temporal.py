"""
Fast Temporal Ticket Assignment Environment.

This environment simulates realistic time-based ticket assignment with agent
capacity constraints and queue management, optimized for fast training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from data.ticket_generator import generate_data, TICKET_TYPES, PRIORITIES


class TemporalTicketAssignmentEnv(gym.Env):
    """
    A fast temporal environment for ticket assignment with realistic time dynamics.

    Key Features:
    - 30 time steps per episode (fast training)
    - Each step = ~10 minutes of simulated time
    - Agents have busy/available states based on resolution times
    - Tickets wait in queue and accumulate penalties
    - New tickets arrive throughout the episode

    State Space
    -----------
    - Current time step (normalized)
    - Queue length (normalized)
    - Agent availability (binary per agent: 0=busy, 1=available)
    - Agent workload (current tickets assigned)
    - Current ticket features (if any waiting)
    - Average queue wait time

    Action Space
    ------------
    - 0 to N-1: Assign current ticket to agent N
    - N: Wait (don't assign yet, useful if better agent will be free soon)

    Reward Function
    ---------------
    - Negative reward for ticket wait time
    - SLA violation penalties (priority-based)
    - Skill matching bonuses
    - Idle agent penalties when queue is full
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_agents: int = 4,
        episode_steps: int = 30,
        tickets_per_step: float = 1.0,
        max_queue_size: int = 20,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Temporal Ticket Assignment Environment.

        Parameters
        ----------
        num_agents : int, default=4
            Number of support agents.
        episode_steps : int, default=30
            Number of time steps per episode.
        tickets_per_step : float, default=1.0
            Average tickets arriving per step.
        max_queue_size : int, default=20
            Maximum queue size before rejecting tickets.
        render_mode : str, optional
            Rendering mode ('human' or None).
        """
        super().__init__()

        self.num_agents = num_agents
        self.episode_steps = episode_steps
        self.tickets_per_step = tickets_per_step
        self.max_queue_size = max_queue_size
        self.render_mode = render_mode

        # Generate all tickets for episode upfront
        total_tickets = int(episode_steps * tickets_per_step * 1.5)  # Buffer
        self.all_tickets_df, self.agents_df = generate_data(total_tickets, num_agents)

        # Episode state
        self.current_step = 0
        self.ticket_queue = deque()  # Waiting tickets
        self.agent_busy_until = np.zeros(num_agents, dtype=np.int32)  # When each agent finishes
        self.agent_workload = np.zeros(num_agents, dtype=np.int32)  # Total tickets assigned
        self.ticket_index = 0  # Next ticket to potentially arrive
        self.completed_tickets = []  # (ticket_id, agent_id, wait_time, resolution_time)
        self.total_wait_time = 0
        self.sla_violations = 0

        # Priority weights for SLA
        self.priority_sla_steps = {
            'Critical': 3,   # Must resolve in 3 steps (30 min)
            'High': 6,       # 6 steps (60 min)
            'Medium': 12,    # 12 steps (120 min)
            'Low': 20        # 20 steps (200 min)
        }

        # Action space: assign to agent 0...N-1, or wait (N)
        self.action_space = spaces.Discrete(num_agents + 1)

        # Observation space
        # [time_step, queue_length, *agent_available, *agent_workload,
        #  ticket_type (5), ticket_priority (4), ticket_complexity, ticket_age]
        obs_dim = 2 + num_agents + num_agents + 5 + 4 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _add_arriving_tickets(self):
        """Add new tickets arriving at this time step."""
        # Poisson-like arrival: random number based on avg rate
        num_arriving = np.random.poisson(self.tickets_per_step)

        for _ in range(num_arriving):
            if self.ticket_index >= len(self.all_tickets_df):
                break

            if len(self.ticket_queue) >= self.max_queue_size:
                break  # Queue full, reject ticket

            ticket = self.all_tickets_df.iloc[self.ticket_index]
            # Add ticket to queue with arrival time
            self.ticket_queue.append({
                'ticket': ticket,
                'arrival_step': self.current_step,
                'ticket_id': ticket['ticket_id']
            })
            self.ticket_index += 1

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Time step (normalized)
        time_norm = self.current_step / self.episode_steps

        # Queue length (normalized)
        queue_norm = len(self.ticket_queue) / self.max_queue_size

        # Agent availability (1 if available now, 0 if busy)
        agent_available = (self.agent_busy_until <= self.current_step).astype(np.float32)

        # Agent workload (normalized)
        max_workload = max(np.max(self.agent_workload), 1)
        workload_norm = self.agent_workload / max_workload

        # Current ticket features (if queue not empty)
        if len(self.ticket_queue) > 0:
            ticket_item = self.ticket_queue[0]
            ticket = ticket_item['ticket']

            # Type one-hot
            type_onehot = np.zeros(len(TICKET_TYPES), dtype=np.float32)
            type_idx = TICKET_TYPES.index(ticket['type'])
            type_onehot[type_idx] = 1.0

            # Priority one-hot
            priority_onehot = np.zeros(len(PRIORITIES), dtype=np.float32)
            priority_idx = PRIORITIES.index(ticket['priority'])
            priority_onehot[priority_idx] = 1.0

            # Complexity (normalized)
            complexity_norm = ticket['complexity'] / 5.0

            # Age in queue (normalized by SLA)
            age_steps = self.current_step - ticket_item['arrival_step']
            sla_threshold = self.priority_sla_steps[ticket['priority']]
            age_norm = min(age_steps / sla_threshold, 1.0)
        else:
            # No tickets in queue - zero features
            type_onehot = np.zeros(len(TICKET_TYPES), dtype=np.float32)
            priority_onehot = np.zeros(len(PRIORITIES), dtype=np.float32)
            complexity_norm = 0.0
            age_norm = 0.0

        # Concatenate observation
        obs = np.concatenate([
            [time_norm],
            [queue_norm],
            agent_available,
            workload_norm,
            type_onehot,
            priority_onehot,
            [complexity_norm],
            [age_norm]
        ]).astype(np.float32)

        return obs

    def _calculate_resolution_steps(self, ticket, agent_idx: int) -> int:
        """Calculate how many steps it takes for agent to resolve ticket."""
        agent = self.agents_df.iloc[agent_idx]

        # Base time from complexity (1-5 complexity = 1-5 steps)
        base_steps = ticket['complexity']

        # Adjust by agent skill (0.5 skill = 2x time, 1.0 skill = 1x time)
        skill = agent['skills'][ticket['type']]
        skill_multiplier = 2.0 - skill  # 0.5->1.5x, 1.0->1.0x

        resolution_steps = int(np.ceil(base_steps * skill_multiplier))
        return max(1, resolution_steps)  # At least 1 step

    def _calculate_reward(
        self,
        action: int,
        ticket_item: Dict,
        agent_idx: int
    ) -> float:
        """Calculate reward for assignment decision."""
        ticket = ticket_item['ticket']
        agent = self.agents_df.iloc[agent_idx]

        # 1. Wait time penalty (negative for each step in queue)
        wait_steps = self.current_step - ticket_item['arrival_step']
        wait_penalty = -wait_steps * 0.5

        # 2. SLA compliance
        sla_threshold = self.priority_sla_steps[ticket['priority']]
        if wait_steps > sla_threshold:
            sla_penalty = -10.0 * (wait_steps - sla_threshold)  # Severe penalty
        else:
            sla_penalty = 0.0

        # 3. Skill match bonus
        skill = agent['skills'][ticket['type']]
        skill_bonus = skill * 5.0  # 0.5->2.5, 1.0->5.0

        # 4. Priority urgency
        priority_weights = {'Low': 1.0, 'Medium': 2.0, 'High': 3.0, 'Critical': 5.0}
        priority_bonus = priority_weights[ticket['priority']]

        # Total reward
        reward = wait_penalty + sla_penalty + skill_bonus + priority_bonus

        return float(reward)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Regenerate tickets and agents
        total_tickets = int(self.episode_steps * self.tickets_per_step * 1.5)
        self.all_tickets_df, self.agents_df = generate_data(total_tickets, self.num_agents)

        # Reset state
        self.current_step = 0
        self.ticket_queue = deque()
        self.agent_busy_until = np.zeros(self.num_agents, dtype=np.int32)
        self.agent_workload = np.zeros(self.num_agents, dtype=np.int32)
        self.ticket_index = 0
        self.completed_tickets = []
        self.total_wait_time = 0
        self.sla_violations = 0

        # Add initial tickets
        self._add_arriving_tickets()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step."""
        reward = 0.0

        # Handle assignment if there's a ticket in queue
        if len(self.ticket_queue) > 0 and action < self.num_agents:
            # Agent assignment action
            agent_idx = action

            # Check if agent is available
            if self.agent_busy_until[agent_idx] <= self.current_step:
                # Assign ticket
                ticket_item = self.ticket_queue.popleft()
                ticket = ticket_item['ticket']

                # Calculate resolution time
                resolution_steps = self._calculate_resolution_steps(ticket, agent_idx)

                # Update agent state
                self.agent_busy_until[agent_idx] = self.current_step + resolution_steps
                self.agent_workload[agent_idx] += 1

                # Calculate reward
                reward = self._calculate_reward(action, ticket_item, agent_idx)

                # Track completion
                wait_time = self.current_step - ticket_item['arrival_step']
                self.total_wait_time += wait_time

                sla_threshold = self.priority_sla_steps[ticket['priority']]
                if wait_time > sla_threshold:
                    self.sla_violations += 1

                self.completed_tickets.append({
                    'ticket_id': ticket_item['ticket_id'],
                    'agent_id': agent_idx,
                    'wait_time': wait_time,
                    'resolution_steps': resolution_steps
                })
            else:
                # Agent busy - small penalty for invalid action
                reward = -2.0

        elif len(self.ticket_queue) > 0 and action == self.num_agents:
            # Wait action - small penalty for delaying
            reward = -0.5

        # Advance time
        self.current_step += 1

        # Add new arriving tickets
        if self.current_step < self.episode_steps:
            self._add_arriving_tickets()

        # Penalty for tickets waiting in queue
        if len(self.ticket_queue) > 0:
            reward -= len(self.ticket_queue) * 0.1

        # Check if episode done
        terminated = (self.current_step >= self.episode_steps and len(self.ticket_queue) == 0)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        avg_wait = self.total_wait_time / len(self.completed_tickets) if self.completed_tickets else 0
        sla_compliance = 1.0 - (self.sla_violations / len(self.completed_tickets)) if self.completed_tickets else 1.0

        return {
            'current_step': self.current_step,
            'queue_length': len(self.ticket_queue),
            'tickets_completed': len(self.completed_tickets),
            'avg_wait_time': avg_wait,
            'sla_compliance': sla_compliance,
            'sla_violations': self.sla_violations,
            'agent_workloads': self.agent_workload.tolist()
        }

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.episode_steps}")
            print(f"Queue Length: {len(self.ticket_queue)}")
            print(f"Tickets Completed: {len(self.completed_tickets)}")
            print(f"Agent Availability: {(self.agent_busy_until <= self.current_step).astype(int)}")
            print(f"Agent Workload: {self.agent_workload}")
            if self.completed_tickets:
                avg_wait = self.total_wait_time / len(self.completed_tickets)
                sla_comp = 1.0 - (self.sla_violations / len(self.completed_tickets))
                print(f"Avg Wait Time: {avg_wait:.2f} steps")
                print(f"SLA Compliance: {sla_comp:.1%}")
            print(f"{'='*60}")

    def close(self):
        """Clean up resources."""
        pass


# Test the environment
if __name__ == "__main__":
    print("=" * 80)
    print("Fast Temporal Ticket Assignment Environment - Test")
    print("=" * 80)
    print()

    # Create environment
    env = TemporalTicketAssignmentEnv(
        num_agents=4,
        episode_steps=30,
        tickets_per_step=1.0,
        render_mode="human"
    )

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

    # Run a full episode with simple policy
    print("Running full episode with greedy policy...")
    obs, info = env.reset(seed=123)
    total_reward = 0

    for step in range(100):  # Max steps
        # Simple policy: assign to first available agent, otherwise wait
        if len(env.ticket_queue) > 0:
            available_agents = np.where(env.agent_busy_until <= env.current_step)[0]
            if len(available_agents) > 0:
                action = available_agents[0]  # First available
            else:
                action = env.num_agents  # Wait
        else:
            action = env.num_agents  # Wait

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0 or terminated:
            env.render()

        if terminated or truncated:
            print(f"\nEpisode finished at step {step+1}!")
            break

    print(f"\nFinal Episode Statistics:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Tickets Completed: {info['tickets_completed']}")
    print(f"  Average Wait Time: {info['avg_wait_time']:.2f} steps")
    print(f"  SLA Compliance: {info['sla_compliance']:.1%}")
    print(f"  SLA Violations: {info['sla_violations']}")
    print(f"  Final Agent Workloads: {info['agent_workloads']}")
    print()

    # Test with SB3 env checker if available
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
    print("Test completed!")
    print("=" * 80)
