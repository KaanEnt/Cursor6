"""
Evaluation metrics for the RL ticket assignment system.

This module provides functions to evaluate and compare different ticket
assignment policies (RL-trained, random baseline, etc.) using various
performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ticket_generator import PRIORITIES


class TicketAssignmentMetrics:
    """
    Metrics calculator for ticket assignment performance evaluation.

    This class computes various metrics to assess how well a ticket
    assignment policy performs, including resolution times, SLA compliance,
    agent workload balance, and skill matching.
    """

    def __init__(
        self,
        sla_thresholds: Dict[str, int] = None
    ):
        """
        Initialize the metrics calculator.

        Parameters
        ----------
        sla_thresholds : dict, optional
            SLA time thresholds (in arbitrary time units) for each priority level.
            Default: {'Low': 100, 'Medium': 50, 'High': 20, 'Critical': 10}
        """
        if sla_thresholds is None:
            self.sla_thresholds = {
                'Low': 100,
                'Medium': 50,
                'High': 20,
                'Critical': 10
            }
        else:
            self.sla_thresholds = sla_thresholds

    def calculate_resolution_time(
        self,
        ticket_df: pd.DataFrame,
        agent_df: pd.DataFrame,
        assignments: List[Tuple[int, int, float]]
    ) -> Dict[str, float]:
        """
        Calculate resolution time statistics.

        Resolution time is estimated based on:
        - Ticket complexity
        - Agent skill level for that ticket type
        - Agent workload at time of assignment

        Parameters
        ----------
        ticket_df : pd.DataFrame
            DataFrame containing ticket information.
        agent_df : pd.DataFrame
            DataFrame containing agent profiles.
        assignments : list of tuples
            List of (ticket_id, agent_id, reward) tuples.

        Returns
        -------
        dict
            Dictionary with resolution time statistics:
            - 'mean': Average resolution time
            - 'median': Median resolution time
            - 'std': Standard deviation
            - 'min': Minimum resolution time
            - 'max': Maximum resolution time
        """
        resolution_times = []
        agent_workload_tracker = {i: 0 for i in range(len(agent_df))}

        for ticket_id, agent_id, _ in assignments:
            # Get ticket and agent info
            ticket = ticket_df[ticket_df['ticket_id'] == ticket_id].iloc[0]
            agent = agent_df.iloc[agent_id]

            # Calculate base resolution time from complexity
            base_time = ticket['complexity'] * 10

            # Adjust by agent skill (higher skill = faster resolution)
            skill = agent['skills'][ticket['type']]
            skill_multiplier = 2.0 - skill  # 0.5 skill -> 1.5x time, 1.0 skill -> 1.0x time

            # Adjust by current workload (more workload = slower)
            workload_multiplier = 1.0 + (agent_workload_tracker[agent_id] * 0.1)

            # Calculate final resolution time
            resolution_time = base_time * skill_multiplier * workload_multiplier
            resolution_times.append(resolution_time)

            # Update workload tracker
            agent_workload_tracker[agent_id] += 1

        resolution_times = np.array(resolution_times)

        return {
            'mean': float(np.mean(resolution_times)),
            'median': float(np.median(resolution_times)),
            'std': float(np.std(resolution_times)),
            'min': float(np.min(resolution_times)),
            'max': float(np.max(resolution_times))
        }

    def calculate_sla_compliance(
        self,
        ticket_df: pd.DataFrame,
        agent_df: pd.DataFrame,
        assignments: List[Tuple[int, int, float]]
    ) -> Dict[str, float]:
        """
        Calculate SLA compliance rate.

        Parameters
        ----------
        ticket_df : pd.DataFrame
            DataFrame containing ticket information.
        agent_df : pd.DataFrame
            DataFrame containing agent profiles.
        assignments : list of tuples
            List of (ticket_id, agent_id, reward) tuples.

        Returns
        -------
        dict
            Dictionary with SLA compliance statistics:
            - 'overall': Overall compliance rate (0-1)
            - 'by_priority': Compliance rate for each priority level
            - 'violations': Number of SLA violations
            - 'total': Total number of tickets
        """
        resolution_times = []
        priorities = []
        agent_workload_tracker = {i: 0 for i in range(len(agent_df))}

        for ticket_id, agent_id, _ in assignments:
            ticket = ticket_df[ticket_df['ticket_id'] == ticket_id].iloc[0]
            agent = agent_df.iloc[agent_id]

            # Calculate resolution time (same as above)
            base_time = ticket['complexity'] * 10
            skill = agent['skills'][ticket['type']]
            skill_multiplier = 2.0 - skill
            workload_multiplier = 1.0 + (agent_workload_tracker[agent_id] * 0.1)
            resolution_time = base_time * skill_multiplier * workload_multiplier

            resolution_times.append(resolution_time)
            priorities.append(ticket['priority'])
            agent_workload_tracker[agent_id] += 1

        # Calculate compliance
        total_tickets = len(resolution_times)
        violations = 0
        by_priority = {p: {'met': 0, 'total': 0} for p in PRIORITIES}

        for res_time, priority in zip(resolution_times, priorities):
            threshold = self.sla_thresholds[priority]
            by_priority[priority]['total'] += 1

            if res_time <= threshold:
                by_priority[priority]['met'] += 1
            else:
                violations += 1

        # Calculate rates
        overall_compliance = 1.0 - (violations / total_tickets) if total_tickets > 0 else 0.0
        priority_compliance = {
            p: by_priority[p]['met'] / by_priority[p]['total'] if by_priority[p]['total'] > 0 else 0.0
            for p in PRIORITIES
        }

        return {
            'overall': float(overall_compliance),
            'by_priority': priority_compliance,
            'violations': int(violations),
            'total': int(total_tickets)
        }

    def calculate_workload_balance(
        self,
        assignments: List[Tuple[int, int, float]],
        num_agents: int
    ) -> Dict[str, float]:
        """
        Calculate agent workload balance metrics.

        Parameters
        ----------
        assignments : list of tuples
            List of (ticket_id, agent_id, reward) tuples.
        num_agents : int
            Total number of agents.

        Returns
        -------
        dict
            Dictionary with workload balance statistics:
            - 'std': Standard deviation of workload distribution
            - 'gini': Gini coefficient (0=perfect equality, 1=perfect inequality)
            - 'max_workload': Maximum tickets assigned to any agent
            - 'min_workload': Minimum tickets assigned to any agent
            - 'workload_distribution': List of workload per agent
        """
        # Count assignments per agent
        agent_workloads = np.zeros(num_agents)
        for _, agent_id, _ in assignments:
            agent_workloads[agent_id] += 1

        # Calculate Gini coefficient
        sorted_workloads = np.sort(agent_workloads)
        n = len(sorted_workloads)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_workloads)) / (n * np.sum(sorted_workloads)) - (n + 1) / n if np.sum(sorted_workloads) > 0 else 0

        return {
            'std': float(np.std(agent_workloads)),
            'gini': float(gini),
            'max_workload': int(np.max(agent_workloads)),
            'min_workload': int(np.min(agent_workloads)),
            'mean_workload': float(np.mean(agent_workloads)),
            'workload_distribution': agent_workloads.tolist()
        }

    def calculate_specialty_match_rate(
        self,
        ticket_df: pd.DataFrame,
        agent_df: pd.DataFrame,
        assignments: List[Tuple[int, int, float]],
        threshold: float = 0.75
    ) -> Dict[str, float]:
        """
        Calculate how often tickets are assigned to agents with high skill in that type.

        Parameters
        ----------
        ticket_df : pd.DataFrame
            DataFrame containing ticket information.
        agent_df : pd.DataFrame
            DataFrame containing agent profiles.
        assignments : list of tuples
            List of (ticket_id, agent_id, reward) tuples.
        threshold : float, default=0.75
            Skill level threshold to consider as a "good match".

        Returns
        -------
        dict
            Dictionary with specialty matching statistics:
            - 'overall_match_rate': Percentage of tickets matched to skilled agents
            - 'avg_skill_level': Average skill level of assigned agents
            - 'by_ticket_type': Match rates broken down by ticket type
        """
        total_matches = 0
        total_tickets = len(assignments)
        skill_levels = []
        type_stats = {ticket_type: {'matches': 0, 'total': 0, 'skills': []}
                      for ticket_type in ticket_df['type'].unique()}

        for ticket_id, agent_id, _ in assignments:
            ticket = ticket_df[ticket_df['ticket_id'] == ticket_id].iloc[0]
            agent = agent_df.iloc[agent_id]

            ticket_type = ticket['type']
            agent_skill = agent['skills'][ticket_type]

            skill_levels.append(agent_skill)
            type_stats[ticket_type]['total'] += 1
            type_stats[ticket_type]['skills'].append(agent_skill)

            if agent_skill >= threshold:
                total_matches += 1
                type_stats[ticket_type]['matches'] += 1

        overall_match_rate = total_matches / total_tickets if total_tickets > 0 else 0.0
        avg_skill = np.mean(skill_levels) if skill_levels else 0.0

        by_type = {
            ticket_type: {
                'match_rate': stats['matches'] / stats['total'] if stats['total'] > 0 else 0.0,
                'avg_skill': np.mean(stats['skills']) if stats['skills'] else 0.0
            }
            for ticket_type, stats in type_stats.items()
        }

        return {
            'overall_match_rate': float(overall_match_rate),
            'avg_skill_level': float(avg_skill),
            'by_ticket_type': by_type
        }

    def evaluate_policy(
        self,
        ticket_df: pd.DataFrame,
        agent_df: pd.DataFrame,
        assignments: List[Tuple[int, int, float]]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a ticket assignment policy.

        Parameters
        ----------
        ticket_df : pd.DataFrame
            DataFrame containing ticket information.
        agent_df : pd.DataFrame
            DataFrame containing agent profiles.
        assignments : list of tuples
            List of (ticket_id, agent_id, reward) tuples from episode.

        Returns
        -------
        dict
            Comprehensive metrics dictionary containing all evaluation metrics.
        """
        num_agents = len(agent_df)

        metrics = {
            'resolution_time': self.calculate_resolution_time(ticket_df, agent_df, assignments),
            'sla_compliance': self.calculate_sla_compliance(ticket_df, agent_df, assignments),
            'workload_balance': self.calculate_workload_balance(assignments, num_agents),
            'specialty_match': self.calculate_specialty_match_rate(ticket_df, agent_df, assignments)
        }

        return metrics


def random_baseline_policy(env, num_episodes: int = 10) -> List[Dict[str, Any]]:
    """
    Run a random assignment baseline policy for comparison.

    Parameters
    ----------
    env : TicketAssignmentEnv
        The ticket assignment environment.
    num_episodes : int, default=10
        Number of episodes to run.

    Returns
    -------
    list of dict
        List of episode results, each containing assignments and metrics.
    """
    results = []
    metrics_calculator = TicketAssignmentMetrics()

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_assignments = []
        done = False

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Get assignments from environment
        assignments = env.assignments

        # Calculate metrics
        episode_metrics = metrics_calculator.evaluate_policy(
            env.tickets_df,
            env.agents_df,
            assignments
        )

        results.append({
            'episode': episode,
            'assignments': assignments,
            'metrics': episode_metrics
        })

    return results


# Test metrics calculation
if __name__ == "__main__":
    from environments.ticket_env import TicketAssignmentEnv

    print("=" * 80)
    print("Metrics Evaluation - Test")
    print("=" * 80)
    print()

    # Create environment
    env = TicketAssignmentEnv(num_agents=5, num_tickets=50)
    metrics_calc = TicketAssignmentMetrics()

    print("Running random baseline policy...")
    results = random_baseline_policy(env, num_episodes=5)
    print(f"Completed {len(results)} episodes")
    print()

    # Aggregate results
    print("Baseline Policy Performance (averaged over 5 episodes):")
    print("-" * 80)

    avg_resolution_time = np.mean([r['metrics']['resolution_time']['mean'] for r in results])
    avg_sla_compliance = np.mean([r['metrics']['sla_compliance']['overall'] for r in results])
    avg_workload_std = np.mean([r['metrics']['workload_balance']['std'] for r in results])
    avg_specialty_match = np.mean([r['metrics']['specialty_match']['overall_match_rate'] for r in results])

    print(f"Resolution Time:")
    print(f"  Mean: {avg_resolution_time:.2f}")
    print(f"  Std: {np.std([r['metrics']['resolution_time']['mean'] for r in results]):.2f}")
    print()

    print(f"SLA Compliance:")
    print(f"  Overall: {avg_sla_compliance:.2%}")
    print()

    print(f"Workload Balance:")
    print(f"  Std Dev: {avg_workload_std:.2f}")
    print(f"  Gini: {np.mean([r['metrics']['workload_balance']['gini'] for r in results]):.3f}")
    print()

    print(f"Specialty Match:")
    print(f"  Match Rate: {avg_specialty_match:.2%}")
    print(f"  Avg Skill: {np.mean([r['metrics']['specialty_match']['avg_skill_level'] for r in results]):.3f}")
    print()

    # Show detailed results for first episode
    print("Detailed Metrics for Episode 0:")
    print("-" * 80)
    first_result = results[0]['metrics']

    print("\nResolution Time:")
    for key, value in first_result['resolution_time'].items():
        print(f"  {key}: {value:.2f}")

    print("\nSLA Compliance by Priority:")
    for priority, rate in first_result['sla_compliance']['by_priority'].items():
        print(f"  {priority}: {rate:.2%}")

    print("\nWorkload Distribution:")
    print(f"  {first_result['workload_balance']['workload_distribution']}")

    print("\nSpecialty Match by Ticket Type:")
    for ticket_type, stats in first_result['specialty_match']['by_ticket_type'].items():
        print(f"  {ticket_type}: {stats['match_rate']:.2%} (avg skill: {stats['avg_skill']:.3f})")

    print()
    print("=" * 80)
    print("Metrics test completed successfully!")
    print("=" * 80)
