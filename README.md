# CARLA Driving RL Agent
A follow up of my *master's thesis project* involving **deep reinforcement learning** to train an **autonomous driving agent**. In particular, the driving agent is trained by using the *Proximal Policy Optimization* algorithm ([PPO](https://arxiv.org/pdf/1707.06347)) within a simulated driving environment provided by the [CARLA](http://carla.org/) simulator ([paper](https://arxiv.org/pdf/1711.03938)). The reinforcement learning phase is organized into increasingly difficult *stages*, following the idea of [Curriculum Learning](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y). 

This work has been accepted at the International Conference on Image Processing ([ICIP 2021](https://www.2021.ieeeicip.org/)). The conference paper is available [here](https://ieeexplore.ieee.org/abstract/document/9506673/).

Requirements, installation instructions, and results are listed below.

---

## Requirements

Software:
- Python 3.7
- CARLA 0.9.9
- Libraries: install from `requirements.txt`

Hardware (minimum):
- CPU: at least quad or octa core.
- GPU: dedicated, with as much memory as possible.
- RAM: at least 16 or 32 Gb.

---

## Installation

Before running any code from this repo you have to:
1. **Clone this repo**: `git clone https://github.com/Luca96/carla-driving-rl-agent.git`
2. **Download CARLA 0.9.9** from their GitHub repo, [here](https://github.com/carla-simulator/carla/releases/tag/0.9.9) 
   where you can find precompiled binaries which are ready-to-use. Refer to [carla-quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/)
   for more information.
3. **Install CARLA Python bindings** in order to be able to manage CARLA from Python code. Open your terminal and type:
   
    * *Windows*: `cd your-path-to-carla/CARLA_0.9.9.4/WindowsNoEditor/PythonAPI/carla/dist/`
    * *Linux*: `cd your-path-to-carla/CARLA_0.9.9.4/PythonAPI/carla/dist/`
    * Extract `carla-0.9.9-py3.7-XXX-amd64.egg` where `XXX` depends on your OS, e.g. `win` for Windows.
    * Create a `setup.py` file within the extracted folder and write the following:
      ```python
      from distutils.core import setup
      
      setup(name='carla',
            version='0.9.9',
            py_modules=['carla']) 
      ```
    * Install via pip: `pip install -e ~/CARLA_0.9.9.4/PythonAPI/carla/dist/carla-0.9.9-py3.7-XXX-amd64`

Before running the repository's code be sure to **start CARLA first**: 
* *Windows*: `your-path-to/CARLA_0.9.9.4/WindowsNoEditor/CarlaUE4.exe`
* *Linux*: `your-path-to/CARLA_0.9.9.4/./CarlaUE4.sh`
* [optional] To use less resources add these flags to the previous command: `-windowed -ResX=32 -ResY=32 --quality-level=Low`.
    For example `./CarlaUE4.sh --quality-level=Low`.

---

## Examples

Show the agent's network architecture (without running CARLA):
```python
from core import CARLAgent, FakeCARLAEnvironment

agent = CARLAgent(FakeCARLAEnvironment(), batch_size=1, log_mode=None)
agent.summary()
```

Play with the CARLA environment (requires running CARLA):
```python
from core import CARLAEnv
from rl import CARLAPlayWrapper
 
# Set `debug=False` if the framerate is very low.
# For better image quality, increase `image_shape` according to your hardware.
env = CARLAEnv(debug=True, window_size=(900, 245), image_shape=(90, 120, 3)) 
CARLAPlayWrapper(env).play()
```

Reinforcement learning example:
```python
from core import learning

learning.stage_s1(episodes=5, timesteps=256, gamma=0.999, lambda_=0.995, save_every='end', stage_name='stage',
                  seed=42, polyak=0.999, aug_intensity=0.0, repeat_action=1, load_full=False)\
        .run2(epochs=10)
```

The complete training procedure is shown in `main.py`. Be aware that each stage can take long time to finish, so *comment what you don't need!*

>NOTE: When loading the agent, e.g. from `stage_s1` to `stege_s2` be sure to "*manually*"" copy and rename the saved agent's weights, otherwise use the same `stage_name` for each stage.

---

## Agent Architecture

The agent leverages the following neural network architecture:

![agent_architecture](src/agent_architecture.png)

* At each timestep $t$ the agent receives an observation $o_t=\{ o_t^1,\ldots,o_t^4 \}$, where each $o_t^i=[\texttt{image},\texttt{road},\texttt{vehicle},\texttt{navigation}]$.
* So, each component of $o_t^i$ is respectively processed by a [ShuffleNet v2](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf), and feed-forward neural networks. Note that layers aren't copied, so the same layers are applied to each $o_t^i$ for which we get four outputs that are aggregated into a single one by Gated Recurrent Units (GRUs).
* The output of each GRU is then concatenated into a single vector, which is linearly combined (i.e. *linear activation function*) into 512 units.
* The output of such operation is the input for both the *value* and *policy* branches. 

For more details refer to `core/networks.py`, in particular to the `dynamics_layers` function and `CARLANetwork` class.

---

## Results

All the experiments were run on a machine with:
- CPU: Intel i9-10980XE 3.00Ghz 18C/36T,
- RAM: 128Gb RAM,
- GPU: Nvidia Quadro RTX 6000 24Gb.

All agents were evaluated on six metrics (*collision rate, similarity, speed, waypoint distance, total reward, and timesteps*), two disjoint weather sets (only one used during training), over all CARLA towns (from `Town01` to `Town10`) but only trained on `Town03`.

![gif-performance](src/gif_agent.gif)

The following table shows the performance of three agents: *curriculum* (C), *standard* (S), and *untrained* (U). The curriculum agent (C) combines PPO with curriculum learning, whereas the standard agent (S) doesn't use any curriculum. Lastly, the untrained agent (U) has the same architecture of the other two but with random weights, so it just provides (non-trivial) baseline performance for comparison purpose.
![performance table](src/absolute_performance.png)

For detailed results over each evaluation scenario, refer to the extensive evaluation table: `src\extensive_evaluation_table`.

---

## Cite this Work

If this work is useful for your own research, please consider citing or mentioning this repository:
```bibtex
@inproceedings{anzalone2021reinforced,
  title={Reinforced Curriculum Learning For Autonomous Driving In Carla},
  author={Anzalone, Luca and Barra, Silvio and Nappi, Michele},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3318--3322},
  year={2021},
  organization={IEEE}
}
```
