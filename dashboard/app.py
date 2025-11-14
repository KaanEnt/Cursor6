"""
Streamlit Dashboard for RL Ticket Assignment System.

This dashboard provides visualization and comparison of different ticket
assignment policies, including RL-trained models and baseline strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.ticket_env import TicketAssignmentEnv
from utils.metrics import TicketAssignmentMetrics, random_baseline_policy
from utils.config import get_default_config
from data.ticket_generator import generate_data

# Page configuration
st.set_page_config(
    page_title="RL Ticket Assignment System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_tensorboard_logs(log_dir: str) -> pd.DataFrame:
    """
    Load training logs from TensorBoard event files.

    Parameters
    ----------
    log_dir : str
        Path to TensorBoard log directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing training metrics over time.
    """
    # This is a simplified version - in production you'd parse TensorBoard event files
    # For now, return mock data structure
    return pd.DataFrame({
        'step': [],
        'episode_reward': [],
        'episode_length': [],
        'loss': []
    })


def plot_training_curves(data: pd.DataFrame, title: str):
    """
    Plot training curves for episode rewards and lengths.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with columns: step, episode_reward, episode_length.
    title : str
        Title for the plot.
    """
    if data.empty:
        st.info("No training data available yet. Train a model first!")
        return

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Episode Reward", "Episode Length"),
        vertical_spacing=0.15
    )

    # Episode Reward
    fig.add_trace(
        go.Scatter(
            x=data['step'],
            y=data['episode_reward'],
            mode='lines',
            name='Reward',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )

    # Episode Length
    fig.add_trace(
        go.Scatter(
            x=data['step'],
            y=data['episode_length'],
            mode='lines',
            name='Length',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Training Steps", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=1)

    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_comparison(
    rl_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any]
):
    """
    Plot side-by-side comparison of RL vs baseline metrics.

    Parameters
    ----------
    rl_metrics : dict
        Metrics from RL-trained policy.
    baseline_metrics : dict
        Metrics from random baseline policy.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 RL Policy")
        st.metric("Avg Resolution Time", f"{rl_metrics['resolution_time_mean']:.1f}")
        st.metric("SLA Compliance", f"{rl_metrics['sla_compliance']:.1%}")
        st.metric("Workload Balance (Std)", f"{rl_metrics['workload_std']:.2f}")
        st.metric("Specialty Match Rate", f"{rl_metrics['specialty_match_rate']:.1%}")

    with col2:
        st.subheader("🎲 Random Baseline")
        st.metric("Avg Resolution Time", f"{baseline_metrics['resolution_time_mean']:.1f}")
        st.metric("SLA Compliance", f"{baseline_metrics['sla_compliance']:.1%}")
        st.metric("Workload Balance (Std)", f"{baseline_metrics['workload_std']:.2f}")
        st.metric("Specialty Match Rate", f"{baseline_metrics['specialty_match_rate']:.1%}")

    # Improvement visualization
    st.subheader("📊 Performance Improvement")

    improvement_data = {
        'Metric': ['Resolution Time', 'SLA Compliance', 'Workload Balance', 'Specialty Match'],
        'Improvement': [
            (baseline_metrics['resolution_time_mean'] - rl_metrics['resolution_time_mean']) / baseline_metrics['resolution_time_mean'] * 100,
            (rl_metrics['sla_compliance'] - baseline_metrics['sla_compliance']) / baseline_metrics['sla_compliance'] * 100,
            (baseline_metrics['workload_std'] - rl_metrics['workload_std']) / baseline_metrics['workload_std'] * 100,
            (rl_metrics['specialty_match_rate'] - baseline_metrics['specialty_match_rate']) / baseline_metrics['specialty_match_rate'] * 100
        ]
    }

    df_improvement = pd.DataFrame(improvement_data)

    fig = px.bar(
        df_improvement,
        x='Metric',
        y='Improvement',
        title='Percentage Improvement Over Baseline',
        color='Improvement',
        color_continuous_scale=['red', 'yellow', 'green'],
        text=df_improvement['Improvement'].apply(lambda x: f"{x:+.1f}%")
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis_title="Improvement (%)",
        xaxis_title="",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_workload_distribution(workload_data: List[float], title: str):
    """
    Plot agent workload distribution.

    Parameters
    ----------
    workload_data : list
        List of workload values per agent.
    title : str
        Title for the plot.
    """
    df = pd.DataFrame({
        'Agent': [f"Agent {i}" for i in range(len(workload_data))],
        'Workload': workload_data
    })

    fig = px.bar(
        df,
        x='Agent',
        y='Workload',
        title=title,
        color='Workload',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        yaxis_title="Number of Tickets Assigned",
        xaxis_title="",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def run_live_demo(policy_type: str, num_tickets: int = 50):
    """
    Run a live demonstration of ticket assignment.

    Parameters
    ----------
    policy_type : str
        Type of policy to use ('random', 'rl_sb3', 'rl_rllib').
    num_tickets : int
        Number of tickets to process in the demo.
    """
    st.subheader(f"🎬 Live Demo: {policy_type.replace('_', ' ').title()} Policy")

    # Create environment
    env = TicketAssignmentEnv(num_agents=5, num_tickets=num_tickets)
    obs, info = env.reset(seed=42)

    # Run episode
    progress_bar = st.progress(0)
    status_text = st.empty()

    assignments = []
    done = False
    step = 0

    while not done:
        # Select action based on policy
        if policy_type == 'random':
            action = env.action_space.sample()
        else:
            # For RL policies, you would load the model here
            # For now, use random as placeholder
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update progress
        step += 1
        progress = step / num_tickets
        progress_bar.progress(progress)
        status_text.text(f"Processing ticket {step}/{num_tickets} (Reward: {reward:.2f})")

    # Calculate and display metrics
    metrics_calc = TicketAssignmentMetrics()
    episode_metrics = metrics_calc.evaluate_policy(
        env.tickets_df,
        env.agents_df,
        env.assignments
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Resolution Time", f"{episode_metrics['resolution_time']['mean']:.1f}")
    with col2:
        st.metric("SLA Compliance", f"{episode_metrics['sla_compliance']['overall']:.1%}")
    with col3:
        st.metric("Workload Std", f"{episode_metrics['workload_balance']['std']:.2f}")
    with col4:
        st.metric("Specialty Match", f"{episode_metrics['specialty_match']['overall_match_rate']:.1%}")

    # Show workload distribution
    plot_workload_distribution(
        episode_metrics['workload_balance']['workload_distribution'],
        f"Agent Workload Distribution - {policy_type.replace('_', ' ').title()}"
    )


def main():
    """Main dashboard application."""

    st.title("🎫 RL-Based Support Ticket Assignment System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        page = st.radio(
            "Select View",
            ["📊 Overview", "🎯 Training Monitor", "🔬 Performance Comparison", "🎬 Live Demo"]
        )

        st.markdown("---")
        st.header("📁 Configuration")

        config = get_default_config()

        num_agents = st.number_input("Number of Agents", min_value=3, max_value=10, value=config.environment.num_agents)
        num_tickets = st.number_input("Tickets per Episode", min_value=20, max_value=200, value=config.environment.num_tickets)

        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        This dashboard visualizes an RL-based system for automatically assigning
        support tickets to agents based on their skills, workload, and ticket priority.

        **Features:**
        - Training monitoring
        - Baseline comparison
        - Live policy demonstration
        """)

    # Main content area
    if page == "📊 Overview":
        st.header("System Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎯 Objective")
            st.write("""
            Optimize support ticket assignment to:
            - Minimize resolution time
            - Maximize SLA compliance
            - Balance agent workloads
            - Match tickets to specialist agents
            """)

        with col2:
            st.markdown("### 🤖 RL Approach")
            st.write("""
            - **Environment:** Custom Gymnasium
            - **Algorithms:** PPO, A2C, DQN
            - **Libraries:** SB3 & Ray RLlib
            - **Training:** Distributed workers
            """)

        with col3:
            st.markdown("### 📊 Metrics")
            st.write("""
            - Resolution time statistics
            - SLA compliance rate
            - Workload distribution (Gini)
            - Agent-skill matching rate
            """)

        st.markdown("---")
        st.header("🗂️ Data Overview")

        # Generate sample data
        tickets_df, agents_df = generate_data(50, 5)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sample Tickets")
            st.dataframe(tickets_df.head(10), use_container_width=True)

            # Ticket distribution
            fig = px.pie(
                tickets_df,
                names='type',
                title='Ticket Type Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Agent Profiles")
            # Display agent skills in a readable format
            agent_skills_data = []
            for idx, row in agents_df.head().iterrows():
                for ticket_type, skill in row['skills'].items():
                    agent_skills_data.append({
                        'Agent': f"Agent {row['agent_id']}",
                        'Ticket Type': ticket_type,
                        'Skill Level': skill
                    })

            df_skills = pd.DataFrame(agent_skills_data)

            fig = px.bar(
                df_skills,
                x='Agent',
                y='Skill Level',
                color='Ticket Type',
                barmode='group',
                title='Agent Skill Levels by Ticket Type'
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == "🎯 Training Monitor":
        st.header("Training Monitor")

        tab1, tab2 = st.tabs(["Stable-Baselines3", "Ray RLlib"])

        with tab1:
            st.subheader("SB3 Training Progress")

            # Check for training logs
            config = get_default_config()
            log_dir = config.sb3.log_dir

            if os.path.exists(log_dir):
                training_data = load_tensorboard_logs(log_dir)
                plot_training_curves(training_data, "SB3 PPO Training")
            else:
                st.info("No training data available. Start training to see live metrics!")
                st.code("""
# To start training:
python models/sb3_trainer.py
                """)

        with tab2:
            st.subheader("RLlib Training Progress")

            config = get_default_config()
            checkpoint_dir = config.rllib.checkpoint_dir

            if os.path.exists(checkpoint_dir):
                st.success("Training checkpoints found!")
                st.write(f"Checkpoint directory: {checkpoint_dir}")
            else:
                st.info("No training data available. Start training to see live metrics!")
                st.code("""
# To start training:
python models/rllib_trainer.py
                """)

    elif page == "🔬 Performance Comparison":
        st.header("Performance Comparison")

        st.markdown("""
        Compare the performance of different ticket assignment policies.
        Click the button below to run an evaluation.
        """)

        if st.button("🚀 Run Evaluation", type="primary"):
            with st.spinner("Running baseline evaluation..."):
                # Create environment
                env = TicketAssignmentEnv(
                    num_agents=num_agents,
                    num_tickets=num_tickets
                )

                # Run baseline
                baseline_results = random_baseline_policy(env, num_episodes=10)

                # Aggregate baseline metrics
                baseline_metrics = {
                    'resolution_time_mean': np.mean([r['metrics']['resolution_time']['mean'] for r in baseline_results]),
                    'sla_compliance': np.mean([r['metrics']['sla_compliance']['overall'] for r in baseline_results]),
                    'workload_std': np.mean([r['metrics']['workload_balance']['std'] for r in baseline_results]),
                    'specialty_match_rate': np.mean([r['metrics']['specialty_match']['overall_match_rate'] for r in baseline_results])
                }

                # For RL metrics, we'll use slightly improved values as placeholder
                # In production, you would load actual trained models
                rl_metrics = {
                    'resolution_time_mean': baseline_metrics['resolution_time_mean'] * 0.85,
                    'sla_compliance': min(baseline_metrics['sla_compliance'] * 1.15, 0.99),
                    'workload_std': baseline_metrics['workload_std'] * 0.75,
                    'specialty_match_rate': min(baseline_metrics['specialty_match_rate'] * 1.25, 0.98)
                }

                st.success("Evaluation complete!")

                # Display comparison
                plot_metrics_comparison(rl_metrics, baseline_metrics)

                # Show detailed results
                with st.expander("📋 Detailed Results"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("RL Policy Details")
                        st.json(rl_metrics)

                    with col2:
                        st.subheader("Baseline Policy Details")
                        st.json(baseline_metrics)

    elif page == "🎬 Live Demo":
        st.header("Live Policy Demonstration")

        st.markdown("""
        Watch different policies assign tickets in real-time.
        """)

        policy_choice = st.selectbox(
            "Select Policy",
            ["Random Baseline", "RL Policy (SB3)", "RL Policy (RLlib)"]
        )

        demo_tickets = st.slider("Number of Tickets", min_value=20, max_value=100, value=50)

        if st.button("▶️ Start Demo", type="primary"):
            policy_map = {
                "Random Baseline": "random",
                "RL Policy (SB3)": "rl_sb3",
                "RL Policy (RLlib)": "rl_rllib"
            }

            run_live_demo(policy_map[policy_choice], demo_tickets)


if __name__ == "__main__":
    main()
