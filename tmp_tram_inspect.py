import numpy as np
from deeptime.markov.msm import TRAM, TRAMDataset

n_therm_states = 3
n_markov_states = 6
trajectory_length = 60
seed=11
rng = np.random.default_rng(seed)

dtrajs = []
bias_matrices = []
base_cycle = np.arange(n_markov_states, dtype=int)
tiled = np.tile(base_cycle, trajectory_length // n_markov_states + 1)
for therm_state in range(n_therm_states):
    noise = rng.integers(0, n_markov_states, size=trajectory_length)
    dtraj = np.mod(tiled[:trajectory_length] + noise + therm_state, n_markov_states)
    dtrajs.append(dtraj.astype(int))
    bias = np.empty((trajectory_length, n_therm_states), dtype=float)
    for other_state in range(n_therm_states):
        penalty = abs(therm_state - other_state)
        bias[:, other_state] = penalty + 0.2 * rng.normal(size=trajectory_length)
    bias_matrices.append(bias)

tram = TRAM(lagtime=2, count_mode="sliding", init_strategy="MBAR")
ds = TRAMDataset(dtrajs=dtrajs, bias_matrices=bias_matrices)
model = tram.fit(ds).fetch_model()
msm_collection = model.msm_collection
print('before select index', msm_collection.current_model)
msm_collection.select(1)
print('after select index', msm_collection.current_model)
print('transition matrix sum rows', np.round(msm_collection.transition_matrix.sum(axis=1), 6))
count_model = msm_collection.count_model
if count_model is not None:
    print('count matrix shape', count_model.count_matrix.shape)
    print('count matrix row sums', np.round(count_model.count_matrix.sum(axis=1)[:3], 3))
else:
    print('count model missing')
