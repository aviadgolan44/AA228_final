from train_iter import TrainIter

"""
epis = [30000, 50000, 60000, 60000, 60000, 60000]
time_steps = [50, 100, 150, 200, 200, 200]
mark_bound = [2, 3, 5, 5, 5, 5]
n_peds = [0, 0, 0, 5, 10, 20]

for i in range(1, len(epis)):
    train = TrainIter(num_epi=epis[i], time_steps=time_steps[i],
                      n_peds=n_peds[i], bound=10,
                      mark_bound=mark_bound[i], cur_iter=i+1)
    print(f"epis: {epis[i]}, n_peds: {n_peds[i]}\n",
          f"time_steps: {time_steps[i]}, mark_bound: {mark_bound[i]}")
    train.run()
"""

train = TrainIter(num_epi=50000, time_steps=150, n_peds=10, bound=5, mark_bound=4, cur_iter=4)
train.run()
