# Differentiable Hybrid Traffic Simulation

This is the code repository for traffic simulation code that was used for our paper, [Differentiable Hybrid Traffic Simulation](https://arxiv.org/abs/2210.08046), which was presented in SIGGRAPH ASIA 2022.
In this repository, we provide scripts for solving inverse problems and intersection signal control problems in the paper. Please refer to the papers for more details.

![alt text](https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/demo_intro.png "Our simulation rendered in Unity Engine")

## Prerequisites

* pytorch
* numpy
* scipy
* tensorboard
* tqdm
* cma
* highway-env
* matplotlib

## Inverse Problem

In this problem, we aim to find initial traffic states that end up in the final states which are given as input.
The optimization graphs below (Macro, Micro, and Hybrid from left to right) show that our gradient based method converges to the better initial states much faster than the other gradient-free methods.

<img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/inverse/macro/end_optimization_graph.png" alt="Macro" width="200"/><img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/inverse/micro/end_optimization_graph.png" alt="Micro" width="200"/>
<img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/inverse/hybrid/end_optimization_graph.png" alt="Hybrid" width="200"/>

Users can run following commands to solve this problem.

```
python example/inverse/macro.py
python example/inverse/micro.py
python example/inverse/hybrid.py 
```

## Intersection Signal Control Problem

In this problem, we aim to optimize the time allocations for traffic signals so that it can minimize the traffic flow that waits for the signal.
For instance, we can optimize the traffic signals to reduce waiting traffic shown in left to right. Note that traffic rendered in red is reduced.

<img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/control/itscp/macro/macro2/epoch_0.gif" alt="Before" width="300"/> <img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/control/itscp/macro/macro2/epoch_100.gif" alt="Before" width="300"/>

The optimization graphs below (Macro, Micro, and Hybrid from left to right) show that our differentiable simulator can be used for this task successfully.

<img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/control/itscp/macro/itscp_optimization_graph.png" alt="Macro" width="200"/> <img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/control/itscp/micro/itscp_optimization_graph.png" alt="Micro" width="200"/> 
<img src="https://github.com/SonSang/diff-hybrid-traffic-sim/blob/master/example/_result/control/itscp/hybrid/itscp_optimization_graph.png" alt="Hybrid" width="200"/>

Users can run following commands to solve this problem.

```
bash ./run_itscp_macro.sh
bash ./run_itscp_micro.sh
bash ./run_itscp_hybrid.sh
```

## Future Works

Currently, we are optimizing the code base to overcome the computational inefficiency of code presented here.
Also, we are working on introducing more diverse set of traffic problems we can solve with our simulator.
Pace car problem presented in our paper is also in the process of refactoring.
Please keep eyes on our future works if you are interested in them!

## Citation

If you refer to our paper, please consider citing our paper with:

```bibtex
@article{son2022differentiable,
  title={Differentiable hybrid traffic simulation},
  author={Son, Sanghyun and Qiao, Yi-Ling and Sewall, Jason and Lin, Ming C},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  pages={1--10},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
