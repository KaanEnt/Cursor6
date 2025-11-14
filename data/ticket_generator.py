"""
Synthetic data generator for support ticket assignment system.

This module provides functionality to generate mock support tickets and agent
profiles for training and testing a reinforcement learning-based ticket
assignment system.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# Constants for ticket attributes
TICKET_TYPES = ['Billing', 'Technical', 'Account', 'Integration', 'Bug Report']
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']


def generate_data(num_tickets: int, num_agents: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic support tickets and agent profiles.

    This function creates two DataFrames: one containing support tickets with
    various attributes, and another containing agent profiles with skill levels
    for each ticket type.

    Parameters
    ----------
    num_tickets : int
        The number of support tickets to generate. Must be non-negative.
    num_agents : int
        The number of agents to generate. Must be non-negative.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - tickets_df: DataFrame with columns ['ticket_id', 'type', 'priority', 'complexity']
        - agents_df: DataFrame with columns ['agent_id', 'skills']

        The 'skills' column contains dictionaries mapping ticket types to skill
        levels (float between 0.5 and 1.0).

    Examples
    --------
    >>> tickets, agents = generate_data(num_tickets=100, num_agents=5)
    >>> print(tickets.head())
    >>> print(agents.head())

    Notes
    -----
    - Ticket IDs are sequential integers starting from 0
    - Agent IDs are sequential integers starting from 0
    - Ticket types are randomly selected from: 'Billing', 'Technical', 'Account',
      'Integration', 'Bug Report'
    - Priorities are randomly selected from: 'Low', 'Medium', 'High', 'Critical'
    - Complexity is an integer between 1 and 5 (inclusive)
    - Agent skill levels are floats between 0.5 and 1.0 for each ticket type
    """
    # Handle edge case: zero tickets
    if num_tickets == 0:
        tickets_df = pd.DataFrame(columns=['ticket_id', 'type', 'priority', 'complexity'])
        tickets_df = tickets_df.astype({
            'ticket_id': 'int64',
            'type': 'object',
            'priority': 'object',
            'complexity': 'int64'
        })
    else:
        # Generate ticket data
        ticket_ids = np.arange(num_tickets)
        types = np.random.choice(TICKET_TYPES, size=num_tickets)
        priorities = np.random.choice(PRIORITIES, size=num_tickets)
        complexities = np.random.randint(1, 6, size=num_tickets)  # 1 to 5 inclusive

        # Create tickets DataFrame
        tickets_df = pd.DataFrame({
            'ticket_id': ticket_ids,
            'type': types,
            'priority': priorities,
            'complexity': complexities
        })

    # Handle edge case: zero agents
    if num_agents == 0:
        agents_df = pd.DataFrame(columns=['agent_id', 'skills'])
        agents_df = agents_df.astype({
            'agent_id': 'int64',
            'skills': 'object'
        })
    else:
        # Generate agent data
        agent_ids = np.arange(num_agents)

        # Generate skills for each agent (dictionary mapping ticket type to skill level)
        agent_skills = []
        for _ in range(num_agents):
            skills = {
                ticket_type: np.random.uniform(0.5, 1.0)
                for ticket_type in TICKET_TYPES
            }
            agent_skills.append(skills)

        # Create agents DataFrame
        agents_df = pd.DataFrame({
            'agent_id': agent_ids,
            'skills': agent_skills
        })

    return tickets_df, agents_df


if __name__ == "__main__":
    """
    Test the data generator by creating a sample dataset and displaying it.
    """
    print("=" * 80)
    print("Support Ticket Assignment System - Data Generator Test")
    print("=" * 80)
    print()

    # Generate sample data
    NUM_TICKETS = 100
    NUM_AGENTS = 5

    print(f"Generating {NUM_TICKETS} tickets and {NUM_AGENTS} agents...")
    print()

    tickets, agents = generate_data(num_tickets=NUM_TICKETS, num_agents=NUM_AGENTS)

    # Display tickets
    print("TICKETS DATASET")
    print("-" * 80)
    print(f"Shape: {tickets.shape}")
    print(f"Columns: {list(tickets.columns)}")
    print()
    print("First 10 tickets:")
    print(tickets.head(10))
    print()
    print("Ticket Statistics:")
    print(f"  Types: {tickets['type'].value_counts().to_dict()}")
    print(f"  Priorities: {tickets['priority'].value_counts().to_dict()}")
    print(f"  Complexity range: {tickets['complexity'].min()} - {tickets['complexity'].max()}")
    print()

    # Display agents
    print("AGENTS DATASET")
    print("-" * 80)
    print(f"Shape: {agents.shape}")
    print(f"Columns: {list(agents.columns)}")
    print()
    print("Agent profiles:")
    for idx, row in agents.iterrows():
        print(f"  Agent {row['agent_id']}:")
        for ticket_type, skill_level in row['skills'].items():
            print(f"    {ticket_type:15s}: {skill_level:.3f}")
        print()

    # Test edge cases
    print("EDGE CASE TESTS")
    print("-" * 80)
    empty_tickets, empty_agents = generate_data(num_tickets=0, num_agents=0)
    print(f"Empty tickets DataFrame shape: {empty_tickets.shape}")
    print(f"Empty tickets columns: {list(empty_tickets.columns)}")
    print(f"Empty agents DataFrame shape: {empty_agents.shape}")
    print(f"Empty agents columns: {list(empty_agents.columns)}")
    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
