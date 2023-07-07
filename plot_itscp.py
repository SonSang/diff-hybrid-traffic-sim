import numpy as np
import os 
import matplotlib.pyplot as plt

path = './result/control/itscp'
mode = 'hybrid'

path0 = []
for file in os.listdir(path):
    n_path = os.path.join(path, file)
    if os.path.isdir(n_path) and mode in file:
        path0.append(n_path)

n_path_list = []
for path1 in path0:
    for file in os.listdir(path1):
        n_path = os.path.join(path1, file)
        if os.path.isdir(n_path):
            n_path_list.append(n_path)

eval_list = []
for n_path in n_path_list:
    with open(os.path.join(n_path, "eval.txt")) as f:
        vals = f.readlines()
        eval = []
        for val in vals:
            eval.append(float(val))
        
        eval_list.append(eval)
        
eval_list = np.array(eval_list)

# plot;

colors = [
    [0, 0.45, 0.74],            # Ours (GD)
]

params = {'legend.fontsize': 25,
        'figure.figsize': (12, 9),
        'axes.labelsize': 30,
        'axes.titlesize': 30,
        'xtick.labelsize':30,
        'ytick.labelsize':30}

plt.rcParams.update(params)
plt.grid(alpha=0.3)

plt.clf()

xx = np.array(list(range(0, len(eval_list[0]))))
if mode == 'macro':
    xx = xx * 10
elif mode == 'micro':
    xx = xx * 20
else:
    xx = xx * 10

mean_ = eval_list.mean(0)
std_ = eval_list.std(0)
color = colors[0]

plt.plot(xx, mean_, color=color, linewidth=4)
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)

plt.legend()
plt.title(f"{mode}")
plt.xlabel("Epoch")
plt.ylabel("Traffic")
#plt.yscale('log')
plt.grid()

plt.savefig("itscp_optimization_graph.png")