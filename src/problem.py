import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification
from scipy.spatial import KDTree


class State:
    """
    State definition to collect all state variables.
    """

    def __init__(self, n_peds, ped_pos, ped_vel,
                 agent_pos, agent_vel, mark_bound, bound):
        # initialize number of pedestrians
        self.n_peds = n_peds

        # initialize ped positions and velocities
        self.ped_pos = ped_pos
        self.ped_vel = ped_vel

        self.ped_tree = KDTree(self.ped_pos)

        # initialize agent position and velocity
        self.agent_pos = agent_pos
        self.agent_vel = agent_vel

        # initialize position of mark
        self.mark = np.random.uniform(-mark_bound, mark_bound, size=(2))

        # set bounds of area
        self.bound = bound

    def state_vec(self):
        """
        Create state vector from ped_pos, agent_pos, and mark
        first entry is distances in x and y from agent to closest ped
        second entry is distance of agent to mark
        """
        num_neighbors = 10
        # get 10 closest within 10 units in order (this is probably enough)
        _, ii = self.ped_tree.query(
            self.agent_pos,
            k=num_neighbors,
            distance_upper_bound=10
        )

        # make sure arrays are of length num_neighbors
        ped_closest_pos = np.vstack((
            self.ped_pos[ii[ii != len(self.ped_pos)]],
            np.zeros((num_neighbors - len(ii[ii != len(self.ped_pos)]), 2))
        ))
        ped_closest_vel = np.vstack((
            self.ped_pos[ii[ii != len(self.ped_vel)]],
            np.zeros((num_neighbors - len(ii[ii != len(self.ped_vel)]), 2))
        ))

        # want to return closest pedestrians to reduce training time
        return np.concatenate((
            (ped_closest_pos - self.agent_pos).flatten(),
            (ped_closest_vel - self.agent_vel).flatten(),
            self.mark - self.agent_pos,
            np.array([0, 0]) - self.agent_vel
        ))

    def reward(self):
        """
        Reward function.
        """
        # initially set 0
        r = 0

        # get distance from mark
        r -= 10 * np.linalg.norm(self.mark - self.agent_pos)

        # get distances of any pedestrians that intersect the agent
        dd, _ = self.ped_tree.query(
            self.agent_pos,
            k=4,
            distance_upper_bound=1
        )
        # multiply by -100
        r -= 100 * float(np.sum(dd[dd != np.inf]))

        # if agent outside of bounds then its no no
        if self.agent_pos[0] <= -self.bound or self.agent_pos[0] >= self.bound:
            r -= 100 * abs(self.agent_pos[0])
        if self.agent_pos[1] <= -self.bound or self.agent_pos[1] >= self.bound:
            r -= 100 * abs(self.agent_pos[1])
        return r


class Problem:
    """
    Creates a Problem.

    Fields:
    time_steps - number of time steps to run
    n_peds - number of pedestrians to simulate
    bound - the max horizontal or vertical distance from origin (grid bounds)
    accel_bound - bounds of acceleration
    num_accels - number of acceleration values for x and y
    mark_bound - standard deviation of where the mark is placed
    """

    def __init__(self, time_steps=1000, n_peds=10, bound=10,
                 accel_bound=0.5, num_accels=11, mark_bound=2):
        self.time_steps = time_steps
        self.n_peds = n_peds
        self.bound = bound
        self.mark_bound = mark_bound

        # time step interval, acceleration factor, and discount factor
        self.dt = 0.1
        self.accel_factor = 1.0

        # bounds of acceleration and number of accel values
        self.accel_bound = accel_bound
        self.num_accels = num_accels

        # initialize trajectories
        self.reset()

    def boundary_accel(self, ped_pos):
        """
        Creates accelerations away from boundaries.

        Inputs:
        ped_pos - pedestrian positions

        Outputs:
        bound_acc - boundary accelerations on each pedestrian
        """
        # boundary accelerations of 0 for every ped
        bound_acc = np.zeros(ped_pos.shape)

        # only create nonzero accelerations for peds that are not on boundary
        nonzeros = ~np.any(abs(ped_pos) == self.bound, axis=1)

        # simplification of unit vectors of gravities from each boundary
        bound_acc[nonzeros] = np.dstack((
            np.abs(ped_pos[nonzeros, 0] + self.bound)**-2
            - np.abs(ped_pos[nonzeros, 0] - self.bound)**-2,
            np.abs(ped_pos[nonzeros, 1] + self.bound)**-2
            - np.abs(ped_pos[nonzeros, 1] - self.bound)**-2
        ))
        return bound_acc

    def neighbor_accel(self, ped_pos, neigh_inds):
        """
        Creates accelerations of neighbors with respect to each point.

        Inputs:
        ped_pos - pedestrian positions
        inds - indices of closest neighbors

        Outputs:
        sum_full_accels - combined accels of closest neighbors for each ped
        """
        # get displacements of each neighbor from closest neighbors
        disps = ped_pos[:, np.newaxis, :] - ped_pos[neigh_inds, :]

        # get distances to closest neighbors
        dists = np.linalg.norm(disps, axis=2)

        # create accelerations from closest neighbors
        full_accels = disps / dists[:, :, np.newaxis]**3

        # only add accelerations of nonzero distances to avoid infinities
        full_accels[np.abs(full_accels) == np.inf] = 0

        return np.sum(full_accels, axis=1)

    def transition(self, save, action):
        """
        Transition the state forward one step.

        Inputs:
        save - boolean to save trajectories
        action - selected action within action space
        """
        # add random noise to velocities
        # TODO add the random noise back in for the pedestrians
        # state.ped_vel += np.random.normal(0, 0.05, size=(n_peds, 2)) * dt
        # add boundaries with an acceleration factor
        self.state.ped_vel += self.accel_factor * self.boundary_accel(
            self.state.ped_pos) * self.dt

        # create KDTree of trajectories to find nearest neighbors
        self.state.ped_tree = KDTree(self.state.ped_pos)

        # get 2 closest neighbors to each pedestrian including self
        dd, neighbor_inds = self.state.ped_tree.query(self.state.ped_pos, k=2)

        # acceleration factor multiplied by 2 to ensure ped collisions dominate
        self.state.ped_vel += 2 * self.accel_factor * self.neighbor_accel(
            self.state.ped_pos,
            neighbor_inds[:, 1:]
        ) * self.dt

        # clip velocities so that pedestrians aren't too fast
        self.state.ped_vel = np.clip(self.state.ped_vel, -1, 1)

        # update ped positions
        self.state.ped_pos += self.state.ped_vel * self.dt

        # update agent vel
        self.state.agent_vel += action * self.dt
        self.state.agent_pos += self.state.agent_vel * self.dt

        # add current ped positions and agent position to trajectories if save
        if save:
            self.trajectories[:, self.cur_step, :] = self.state.ped_pos
            self.agent_trajectory[self.cur_step, :] = self.state.agent_pos

    def reset(self):
        """
        Reinitialize problem.

        Inputs:
        accel_bound - maximum absolute value of acceleration
        num_accels - number of acceleration values for each x and y
        """
        np.random.seed()

        # we discretize our action space
        # possible action values with max and min vals and number of accel vals
        self.a_vals = np.linspace(
            -self.accel_bound,
            self.accel_bound,
            self.num_accels
        )

        # initialize space for trajectories
        self.trajectories = np.zeros((self.n_peds, self.time_steps + 1, 2))
        # uniformly distributed initial positions
        self.trajectories[:, 0, :] = np.random.uniform(
            low=[-self.bound, -self.bound],
            high=[self.bound, self.bound],
            size=(self.n_peds, 2)
        )

        # random initial velocities (gaussian distribution)
        mean_vel = [0.0, 0.0]
        std_vel = [1.0, 1.0]
        self.velocities = np.random.normal(
            mean_vel,
            std_vel,
            size=(self.n_peds, 2)
        )

        # initialize state, pass values of trajectories to avoid mutating data
        self.state = State(
            self.n_peds,
            self.trajectories[:, 0, :].copy(),
            self.velocities,
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            self.mark_bound,
            self.bound
        )

        # initialize agent trajectory
        self.agent_trajectory = np.zeros((self.time_steps + 1, 2))
        self.agent_velocity = np.zeros(2)

        # set current step
        self.cur_step = 0

    def step(self, save, action_ind):
        self.cur_step += 1

        # transition to next state
        self.transition(
            save,
            np.array([
                self.a_vals[action_ind[0]],
                self.a_vals[action_ind[1]]
            ])
        )

        # calculate reward
        r = self.state.reward()

        # if agent is near mark or past time_steps then finish
        done = np.linalg.norm(
            self.state.mark - self.state.agent_pos
        ) < 0.2 or self.cur_step >= self.time_steps

        return np.array([self.state.state_vec()], dtype=np.float32), r, done

    def render(self):
        print(f"State: {self.state}, Steps: {self.current_step}")

    def visualize(self):
        """
        Plot trajectories of agent and pedestrians of problem.
        """
        # For visualisation: Plot static trajectories
        plt.figure(figsize=(8, 8))
        for i in range(self.n_peds):
            plt.plot(
                self.trajectories[i, :self.cur_step, 0],
                self.trajectories[i, :self.cur_step, 1],
                alpha=0.7
            )
        plt.plot(
            self.agent_trajectory[:self.cur_step, 0],
            self.agent_trajectory[:self.cur_step, 1],
            label="Agent",
            marker="^",
            color="red"
        )
        plt.plot(
            self.state.mark[0],
            self.state.mark[1],
            label="Mark",
            marker="+",
            color="blue"
        )
        # plt.title("Pedestrian Trajectories with Controlled Agent")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.axis([-self.bound, self.bound, -self.bound, self.bound])
        plt.show()

        # Animation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)

        # Lines for pedestrians and controlled agent
        lines = [ax.plot([], [], marker="o")[0] for _ in range(self.n_peds)]
        controlled_line, = ax.plot(
            [],
            [],
            marker="^",
            color="red",
            markersize=10,
            label="Agent"
        )
        marker_line, = ax.plot(
            [],
            [],
            marker="+",
            color="blue",
            markersize=10,
            label="Pedestrian"
        )

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(
                    [self.trajectories[i, frame, 0]],
                    [self.trajectories[i, frame, 1]]
                )
            controlled_line.set_data(
                [self.agent_trajectory[frame, 0]],
                [self.agent_trajectory[frame, 1]]
            )
            marker_line.set_data([self.state.mark[0]], [self.state.mark[1]])
            return lines + [controlled_line] + [marker_line]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=self.cur_step,
            interval=100,
            blit=True
        )
        plt.legend()
        plt.show()

    def choose_action(self, dqn):
        with torch.no_grad():
            state_tensor = torch.tensor(
                self.state.state_vec(),
                dtype=torch.float32
            ).unsqueeze(0)
            return (
                    dqn(state_tensor)[0].argmax(dim=-1).item(),
                    dqn(state_tensor)[1].argmax(dim=-1).item()
            )

    def simulate(self, dqn=None):
        """
        Simulates trajectories from deep Q nn.

        Inputs:
        dqn - deep Q learning neural network
        """
        self.reset()
        tot_rew = 0

        for t in range(self.time_steps):
            if dqn:
                action = self.choose_action(dqn)
            else:
                action = np.random.randint(0, self.num_accels, size=(2))
            _, r, done = self.step(True, action)
            tot_rew += r
            if done:
                break
        return tot_rew
