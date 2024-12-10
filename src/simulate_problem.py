import torch
from dqn import DQN
from problem import Problem

problem = Problem(time_steps=200, n_peds=10, mark_bound=4, bound=5)
state_dim = len(problem.state.state_vec())
n_actions = len(problem.a_vals)

# load previous model weights
model = DQN(state_dim, n_actions)
model.load_state_dict(torch.load(
    "run/policy_nn_state_dict4.pt",
    weights_only=True
))

rewards = []
for i in range(1):
    problem.reset()
    # simulate trajectories
    tot_rew = problem.simulate(model)
    rewards.append(tot_rew)
# print(sum(rewards) / len(rewards))

# graph results
problem.visualize()
