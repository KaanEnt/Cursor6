import streamlit as st
import plotly.graph_objects as go
import time
import random

st.set_page_config(page_title="Support Agent Arena", layout="wide")

st.title("🎮 Support Agent Battle Arena")

col1, col2, col3, col4 = st.columns(4)

agents = ["Agent 1", "Agent 2", "Agent 3", "Agent 4"]
colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a8dadc']

agent_scores = [0, 0, 0, 0]
agent_workloads = [5, 5, 5, 5]
score_history = {agent: [] for agent in agents}

score_placeholders = []
workload_placeholders = []

for i, (col, agent, color) in enumerate(zip([col1, col2, col3, col4], agents, colors)):
    with col:
        st.markdown(f"### {agent}")
        score_placeholders.append(st.empty())
        workload_placeholders.append(st.empty())

score_chart = st.empty()

for step in range(1000):
    score_delta = [random.randint(1, 10) for _ in range(4)]
    
    for i in range(4):
        agent_scores[i] += score_delta[i]
        agent_workloads[i] = random.randint(1, 10)
        score_history[agents[i]].append(agent_scores[i])
        
        score_placeholders[i].metric("Score", f"{agent_scores[i]}", f"+{score_delta[i]}")
        workload_placeholders[i].progress(agent_workloads[i] / 10)

    fig = go.Figure()
    for i, agent in enumerate(agents):
        fig.add_trace(go.Scatter(
            y=score_history[agent],
            name=agent,
            line=dict(color=colors[i], width=3)
        ))

    fig.update_layout(
        title="Live Score Tracking",
        height=400,
        template="plotly_dark"
    )
    score_chart.plotly_chart(fig, use_container_width=True)

    time.sleep(0.1)

