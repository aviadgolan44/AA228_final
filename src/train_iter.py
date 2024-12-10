import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN
from tqdm import tqdm
from problem import Problem
from collections import namedtuple, deque

# transition variable
Transition = namedtuple(
    'Transition',
    ('state', 'action_x', 'action_y', 'next_state', 'reward')
)


class ReplayMemory:
    """
    Create memory of past transitions to batch sample from.

    Fields:
    capacity - how many past transitions to store
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TrainIter:
    """
    Create training instance.

    Fields:
    device - device to compute on
    num_epi - number of episodes to run
    time_steps - number of time steps that the problem runs for
    n_peds - number of pedestrians in problem
    bound - position bounds of problem
    accel_bound - bounds of acceleration
    num_accels - number of possible acceleration values for x and y
    mark_bound - bounds of mark position stochasticity
    cur_iter - whether to load previous neural network
    """

    def __init__(self, device=torch.device("cpu"), num_epi=10000,
                 time_steps=50, n_peds=20, bound=10, accel_bound=0.5,
                 num_accels=11, mark_bound=2, cur_iter=0):
        self.device = device
        self.num_epi = num_epi

        # predetermined parameters
        self.gam = 0.95
        self.target_update = 20
        self.batch_size = 128

        # eps values for decay and initialization
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = int(0.5 * num_epi)

        # replay memory with capacity
        mem_cap = 10000
        self.mem = ReplayMemory(mem_cap)

        # initialize problem
        self.problem = Problem(
            time_steps=time_steps,
            n_peds=n_peds,
            bound=bound,
            accel_bound=accel_bound,
            num_accels=num_accels,
            mark_bound=mark_bound
        )

        # number of state values and action outputs for dqn
        state_dim = len(self.problem.state.state_vec())
        n_actions = len(self.problem.a_vals)

        # create policy nn and target nn
        self.policy_nn = DQN(state_dim, n_actions).to(self.device)
        self.target_nn = DQN(state_dim, n_actions).to(self.device)

        # set nn to prev weights and vals if loading previous
        self.cur_iter = cur_iter
        if self.cur_iter > 0:
            self.policy_nn.load_state_dict(
                torch.load(
                    f"run/policy_nn_state_dict{self.cur_iter - 1}.pt",
                    weights_only=True
                )
            )

        # copy policy_nn into target_nn
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.target_nn.eval()

        # create adam optimizer using policy nn params
        learning_rate = 1e-4
        self.optimizer = optim.Adam(
            self.policy_nn.parameters(),
            lr=learning_rate
        )

    def choose_action(self, eps):
        """
        Choose an action index for x and y.

        Inputs:
        eps - epsilon value to determine if explore or exploit
        """
        # check if explore or exploit
        if random.random() < eps:
            # explore: choose random action
            return (random.choice(range(11)), random.choice(range(11)))
        else:
            # exploit: choose action with the highest Q-value
            with torch.no_grad():
                state_tensor = torch.tensor(
                    self.problem.state.state_vec(),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                return (
                        self.policy_nn(state_tensor)[0].argmax(dim=-1).item(),
                        self.policy_nn(state_tensor)[1].argmax(dim=-1).item()
                )

    def optimize_model(self):
        """
        Optimize neural networks by using value function.
        """
        # return if not enough mem for batching
        if len(self.mem) < self.batch_size:
            return

        # get transitions from memory
        transitions = self.mem.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # create batches for computation
        state_vec_batch = torch.tensor(
            np.vstack(batch.state),
            dtype=torch.float32
        ).to(self.device)
        action_batch_x = torch.tensor(
            batch.action_x,
            dtype=torch.long
        ).unsqueeze(1).to(self.device)
        action_batch_y = torch.tensor(
            batch.action_y,
            dtype=torch.long
        ).unsqueeze(1).to(self.device)
        rew_batch = torch.tensor(
            batch.reward,
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            dtype=torch.bool,
            device=self.device
        ).to(self.device)
        non_final_next_states = torch.tensor(
            np.vstack([s for s in batch.next_state if s is not None]),
            dtype=torch.float32
        ).to(self.device)

        # Compute Q(s, a)
        q_vals = self.policy_nn(state_vec_batch)
        state_action_vals_x = q_vals[0].gather(1, action_batch_x)
        state_action_vals_y = q_vals[1].gather(1, action_batch_y)

        # Compute target Q-values
        next_state_vals_x = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )
        next_state_vals_y = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        next_q_vals = self.target_nn(
            non_final_next_states
        )
        next_state_vals_x[non_final_mask] = next_q_vals[0].max(1)[0].detach()
        next_state_vals_y[non_final_mask] = next_q_vals[1].max(1)[0].detach()

        target_vals_x = rew_batch + (self.gam * next_state_vals_x.unsqueeze(1))
        target_vals_y = rew_batch + (self.gam * next_state_vals_y.unsqueeze(1))

        # Compute loss and update (add x and y loss together)
        criterion = nn.MSELoss()
        loss_x = criterion(state_action_vals_x, target_vals_x)
        loss_y = criterion(state_action_vals_y, target_vals_y)
        loss = loss_x + loss_y
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        print("Running on device:", self.device)

        # open a file to write reward
        rew_file = open('run/rewards.txt', 'w')

        # training
        for epi in tqdm(range(self.num_epi), desc="Training Episodes"):
            # reset problem and reward
            self.problem.reset()
            tot_rew = 0

            # simulate problem
            for t in range(self.problem.time_steps):
                # epsilon decay
                eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
                    -1.0 * epi / self.eps_decay
                )

                # choose action from action space
                action = self.choose_action(eps)

                # store current state vector
                state_vec = self.problem.state.state_vec()

                # transition to next state
                next_state_vec, rew, done = self.problem.step(
                    save=False, action_ind=action)
                tot_rew += rew

                # store transition in memory
                self.mem.push(
                    state_vec,
                    action[0],
                    action[1],
                    next_state_vec if not done else None,
                    rew
                )

                # optimize model
                self.optimize_model()

                if done:
                    break

            # update target nn to previous policy nn
            if epi % self.target_update == 0:
                self.target_nn.load_state_dict(self.policy_nn.state_dict())

            info = (
                    f"Episode {epi + 1}, "
                    f"Total Reward: {tot_rew}\n"
                    f"Final Pos: {self.problem.state.agent_pos}, "
                    f"Mark Position: {self.problem.state.mark}\n"
                    f"(dx,dy) = ({self.problem.state.state_vec()[-4]}, "
                    f"{self.problem.state.state_vec()[-3]})\n"
            )
            rew_file.write(info + "\n")

        torch.save(
            self.policy_nn.state_dict(),
            f"run/policy_nn_state_dict{self.cur_iter}.pt"
        )
        rew_file.close()


# run sample trainiter
if __name__ == "__main__":
    train_iter = TrainIter(n_peds=0)
    train_iter.run()
