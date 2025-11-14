"""
Simple Demo: Quick Start for RL Ticket Assignment

This script provides a fast, minimal example using a simplified environment.
Perfect for understanding the basics before diving into the full system.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environments.ticket_env import TicketAssignmentEnv
from data.ticket_generator import generate_data


def main():
    print("=" * 80)
    print("🎫 RL Ticket Assignment - Simple Demo")
    print("=" * 80)
    print()

    # --- 1. Generate Agent Skills ---
    print("📊 Step 1: Generating agent skill data...")
    _, agents_df = generate_data(num_tickets=10, num_agents=5)

    # Convert to numpy array for easy display
    agent_skills = np.array([list(agent['skills'].values()) for agent in agents_df['skills']])

    print(f"   Created {len(agent_skills)} agents with skills for 5 ticket types")
    print(f"   Agent 0 skills: {agent_skills[0]}")
    print()

    # --- 2. Create Simple Environment ---
    print("🏗️  Step 2: Creating environment...")
    env = TicketAssignmentEnv(
        num_agents=5,
        num_tickets=50,  # Short episode for demo
        max_agent_load=10
    )

    print(f"   Environment: {env}")
    print(f"   Action space: {env.action_space} (assign to agent 0-4)")
    print(f"   Observation space: {env.observation_space.shape}")
    print()

    # --- 3. Validate Environment ---
    print("✅ Step 3: Validating environment...")
    try:
        check_env(env, warn=True)
        print("   Environment passed all checks!")
    except Exception as e:
        print(f"   Warning: {e}")
    print()

    # --- 4. Test Random Policy ---
    print("🎲 Step 4: Testing random assignment policy...")
    obs, info = env.reset(seed=42)
    total_reward = 0
    step_count = 0

    while True:
        # Random action: pick random agent
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    print(f"   Episode completed in {step_count} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward: {total_reward/step_count:.2f}")
    print(f"   Final workloads: {info['agent_workloads']}")
    print()

    # --- 5. Train RL Agent (Fast) ---
    print("🤖 Step 5: Training RL agent (PPO, 10K steps)...")
    print("   This will take ~30-60 seconds...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=0
    )

    # Train for just 10K steps (very fast)
    model.learn(total_timesteps=10_000, progress_bar=True)
    print("   Training complete!")
    print()

    # --- 6. Test Trained Policy ---
    print("🎯 Step 6: Testing trained RL policy...")
    obs, info = env.reset(seed=42)
    trained_reward = 0
    trained_steps = 0

    while True:
        # Use trained model to pick action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        trained_reward += reward
        trained_steps += 1

        if terminated or truncated:
            break

    print(f"   Episode completed in {trained_steps} steps")
    print(f"   Total reward: {trained_reward:.2f}")
    print(f"   Average reward: {trained_reward/trained_steps:.2f}")
    print(f"   Final workloads: {info['agent_workloads']}")
    print()

    # --- 7. Compare Results ---
    print("📊 Step 7: Comparison")
    print("-" * 80)
    improvement = ((trained_reward - total_reward) / abs(total_reward)) * 100
    print(f"   Random Policy:  {total_reward:.2f} average reward")
    print(f"   Trained Policy: {trained_reward:.2f} average reward")
    print(f"   Improvement:    {improvement:+.1f}%")
    print()

    print("=" * 80)
    print("✅ Demo Complete!")
    print()
    print("Next steps:")
    print("  1. Try the temporal environment: python environments/ticket_env_temporal.py")
    print("  2. Run full training: python models/sb3_trainer.py")
    print("  3. Launch dashboard: streamlit run dashboard/app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
