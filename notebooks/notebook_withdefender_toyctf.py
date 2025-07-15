# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Attacker agent benchmark comparison in presence of a basic defender (ToyCTF)

This notebook can be run directly from VSCode, or exported as a Jupyter Notebook.
"""

# %%
import sys
import os
import logging
import gymnasium as gym
import importlib
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines
from cyberbattle._env.cyberbattle_env import AttackerGoal, DefenderConstraint, CyberBattleEnv

importlib.reload(learner)
importlib.reload(p)
importlib.reload(w)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")
# %matplotlib inline

# %% {"tags": ["parameters"]}
iteration_count = 600
training_episode_count = 10
plots_dir = "output/plots"

# %%
gym_env = gym.make(
    "CyberBattleToyCtf-v0",
    attacker_goal=AttackerGoal(own_atleast=0, own_atleast_percent=1.0),
    defender_constraint=DefenderConstraint(maintain_sla=0.80),
    defender_agent=ScanAndReimageCompromisedMachines(probability=0.6, scan_capacity=2, scan_frequency=5),
).unwrapped

cyberbattletoyctf_defender = gym_env.unwrapped
assert isinstance(cyberbattletoyctf_defender, CyberBattleEnv)

# Adjust these bounds if needed for ToyCTF
ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=10, maximum_node_count=10, identifiers=cyberbattletoyctf_defender.identifiers)

# %%
dqn_with_defender = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattletoyctf_defender,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(ep=ep, gamma=0.15, replay_memory_size=10000, target_update=5, batch_size=256, learning_rate=0.01),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DQL",
)

# %%
dql_exploit_run = learner.epsilon_greedy_search(
    cyberbattletoyctf_defender,
    ep,
    learner=dqn_with_defender["learner"],
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    render=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting DQL",
)

# %%
credlookup_run = learner.epsilon_greedy_search(
    cyberbattletoyctf_defender,
    ep,
    learner=rca.CredentialCacheExploiter(),
    episode_count=10,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=10000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Credential lookups (ϵ-greedy)",
)

# %%
# Plots
all_runs = [credlookup_run, dqn_with_defender, dql_exploit_run]
p.plot_averaged_cummulative_rewards(
    all_runs=all_runs,
    title=f"Attacker agents vs Basic Defender -- rewards\n env={cyberbattletoyctf_defender.name}, episodes={training_episode_count}",
    save_at=os.path.join(plots_dir, "withdefender-toyctf-cumreward.png")
)
p.plot_averaged_availability(
    title=f"Attacker agents vs Basic Defender -- availability\n env={cyberbattletoyctf_defender.name}, episodes={training_episode_count}",
    all_runs=all_runs,
    show=False
)
