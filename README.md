<div align = "center">
<img src="images/cameleon.png" alt="CAMeLeon"  border=0>
</div>

<div align="center">
<!-- Python -->
<img src = "https://img.shields.io/badge/python-v3.6+-blue.svg"
         alt = "Python" />
<!-- Release -->
    <img src = "https://img.shields.io/badge/Release-1.0.0-00CC33.svg" 
         alt = "Release" />
<!-- Build Status -->
    <img src = "https://img.shields.io/badge/Build-failing-FF0000.svg" 
         alt = "Build Status" />   
<!-- Development Status -->
    <img src = "https://img.shields.io/badge/Development-in%20progress-FF9933.svg" 
         alt = "Development Status" /> 
<!-- Stability -->
    <img src = "https://img.shields.io/badge/Stability-experimental-FF9933.svg" 
         alt = "Stability" />
</div>

<div align="center">
  <sub>Built with :heart: &nbsp; by 
    <a href = "https://samshowalter.github.io"> Sam Showalter</a>
  </sub>
</div> 
<br/>


# Contents
- [Overview](#overview)
- [Installation](#install)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Scripts](#scripts)
    + [Environment Development](#manual_control)
    + [Environment Benchmarking](#benchmark)
    + [Agent Training](#agent_training)
    + [Agent Rollouts](#agent_rollouts)
- [Artifacts](#artifacts)
- [Cameleon Environments](#environments)
    + [Canniballs](#canniballs)
- [Known Issues](#known_issues)
- [Usage and Debugging Tips](#tips)
- [Support](#support)


<a name = "overview"></a>
# Overview


> `CAMeLeon` is a simple, flexible interface to apply CAML/CARLI to and train agents on arbitrary RL environments

It's ultimate goals are as follows:
1. Provide a general grid-based interface upon which gridworld games of arbitrary complexity may be built.
2. Seamless integrate the general Cameleon environment interface with RLlib's distributed, high optimized training and rollout system 
3. Ensure that arbitrary model-free (and eventually, model-based) RL algorithms may be trained on Cameleon environments through a standard CLI / API.
4. Effectively rollout trained RL agents in environments while extensively collecting and storing their internal state through a standardized policy extractor
5. Automatically Cameleon rollout information into CAML's **Interestingness-xdrl** and **Imago** imagination evaluation packages.

<a name = "install"></a>
# Installation (setup.py not yet complete!)

```
git clone https://gitlab.sri.com/caml/CAMeLeon.git
cd cameleon
pip install -e .[caml]
```

<a name = "dependencies"></a>
# Dependencies

- `gym-minigrid`
- `argparse`
- `numpy`
- `gym`
- `ray[rllib]`
- `tensorflow` or `pytorch`
- `tqdm`
- `logging`
- `jupyter` (to examine notebooks)
- `matplotlib`
- `pillow`

<a name = "usage"></a>
# Usage

Cameleon is structured to allow for a user to completely define a scenario in which they would like to assess competency of an agent. Therefore, it includes functionality to assist in grid-world game development, benchmarking of created environments (speed, memory, etc.), agent training within an environment, and automated rollouts of a trained agent in the environment for later use in CAML assessments. Below are the scripts, in topological order, that a user would use to construct a Cameleon system.

<a name = "scripts"></a>
# Scripts

In all of the following bash scripts, reference to `$Variables` indicates a user-defined input parameter.

<a name = "manual_control"></a>
## Environment Development 

Building a grid-world environment is relatively straightforward and provides enough flexibility while remaining performant for fast RL training. To assist in this development, Cameleon provides a manual control API to run the environment manually and test its functionality. Relevant artifacts like rewards, observations, and actions can be viewed from the console as well to confirm the agent's interaction with the environment is correct. Once the environment is created, call it from the `env_name` parameter and specify the relevant `key_handler` between minigrid and cameleon. 

```bash
# Run the script
python -m cameleon.bin.manual_control \
  --env_name=$ENV_NAME \
  --key_handler=$KEY_HANDLER \
  --seed=$SEED \
  --tile_size=$TILE_SIZE \
  --verbose=$VERBOSE
```

<a name = "benchmark"></a>
## Environment Benchmarking

After the environment is built and debugged, it is important to verify that during training the environment can cycle at a high speed. This benchmarking script runs this test and several others, including how fast the environment can reset, how fast the encoded frames cycle, and how fast the rendered frames cycle. For the encoding cycle tests, add any relevant wrappers as a comma delimited string to ensure a realistic test.

```bash
# Run the script for env benchmarking
python -m cameleon.bin.benchmark \
  --env_name=$ENV_NAME \
  --wrappers=$WRAPPERS \
  --num_enc_frames=$NUM_ENC_FRAMES \
  --num_viz_frames=$NUM_VIZ_FRAMES \
  --num_resets=$NUM_RESETS \
  --visual=$VISUAL
```

<a name = "agent_training"></a>
## Agent Training

Tapping into the RLlib training API, this script trains an RLlib agent with a few added arguments specific to the Cameleon environment. Specifically, the wrappers argument appropriately wraps the full-observability state to create a state of partial observability or an alternative representation altogether. Beyond some small setup and integration details, this represents the existing RLlib training API, documentation for which can be found [here](https://docs.ray.io/en/master/rllib-training.html).

```bash
# Run the script for training
python -m cameleon.bin.train \
  --env_name=$ENV_NAME \
  --model=$MODEL \
  --wrappers=$WRAPPERS \
  --num_workers=$NUM_WORKERS \
  --num_gpus=$NUM_GPUS \
  --model_dir=$MODEL_DIR \
  --outdir=$OUTPUT_DIR \
  --num_epochs=$NUM_EPOCHS \
  --checkpoint_epochs=$CHECKPOINT_EPOCHS \
  --framework=$FRAMEWORK \
  --config=$CONFIG \
  --verbose=$VERBOSE 
```


<a name = "agent_rollouts"></a>
## Agent Rollouts

Similar to the training API, the rollout API for Cameleon allows the user to set a train agent in an environment and record its behavior both as an encoded observation and, if needed, as a video recording. If the rollout is to be completed with the same setup in which training is conducted, no configuration file needs to be provided. **Important** - Currently setting `--no-frame=True` will incur a significant memory overhead. It is set off by default; use with caution. Beyond this, this script loosely resembles the native RLlib rollout API, for which documentation can be found [here](https://docs.ray.io/en/master/rllib-training.html)

```bash
# Run the script for rollouts
python -m cameleon.bin.rollout \
  --env=$ENV \
  --wrappers=$WRAPPERS \
  --run=$MODEL \
  --num_workers=$NUM_WORKERS \
  --num_gpus=$NUM_GPUS \
  --out=$MODEL \
  --checkpoint=$MODEL_CHECKPOINT \
  --episodes=$EPISODES \
  --steps=$STEPS \
  --out=$OUTPUT_DIR \
  --framework=$FRAMEWORK \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --no-frame=$NO_FRAME \
  --use-hickle=$USE_HICKLE \
  --config=$CONFIG 
```


<a name = "artifacts"></a>
# Cameleon Artifacts

There are several important artifacts that Cameleon generates and automatically stores. These destinations are listed below and pre-configured but can be overridden if necessary.

- **models**: Stores all artifacts from RLlib training, including the configuration file, checkpoints, and tensorboard output for visualizing training. Each run is uniquely identified based on its environment, model, and the current date. Training runs of the exact same configuration on the same day will be overwritten if the previous model checkpoint is not fed to the run as a starting point. If it is, the same directory will be used but there will be no checkpoint collisions.

- **rollouts**: Stores rollout artifacts for trained agent interactions within the environment. The agent's state, observation, action, reward, and ancillary information are tracked for each timestep as a JSON-style python dictionary and then compressed with `pickle`. These rollouts save as separate pickle files to a single, programmatically defined directory and are named based on their process ID (`pid`) and CPU core-specific episode number since rollouts are executed in a distributed setting. If requested, each rollout will have a corresponding video. If video recording is specified, each rollout will automatically generate a folder. Inside, a pickle file of the rollout and a `.mp4` file of the same name will be present.

- **data**: The data file corresponds to any additional features needed specifically for RL competency analysis. These may include extracted features, artifacts derived from an agent rollout (action distribution, value function estimates, etc.). Accordingly, there are data subfolders built for specific purposes. These include:
  + **Interestingness**: Interestingness objects (Agent, Environment) to assess notable moments in agent's trajectory


<a name = "environments"></a>
# Cameleon Environments

Provided with Cameleon are environments. Derived from the basic functionality of [Gym MiniGrid](https://github.com/maximecb/gym-minigrid), Cameleon's
base environment and package structure is built to accommodate highly flexible, configurable environments.
Emphasis is placed on ensuring modularity so that the environment can be tuned precisely for specific scenarios
that facilitate experimentation on RL competency awareness. Information about the dynamics of these environments
is added below.

<a name = "canniballs"></a>
## 
<div align = "center">
<img src="images/canniballs_game.png" alt="Canniballs"  border=0>
</div>

Canniballs is a simple grid-world game built to examine agent competency in a highly stochastic environment with subgoals. There are several types of objects with which the agent can interact. Details on these objects is included below. The overall goal of the game is to eat the other ball (canniball) objects present on the screen. To do this, the agent must increase its power score such that it exceeds its opponent. It can do this by finding and consuming food. Once the opponent's score exceeds an opponent, it can consume it and use it as food as well. The game terminates when the agent is "canniballized", when the agent consumes all other opponents (food constantly regenerates and is not considered), or the environment reaches its step limit. All dynamics in this game are completely configurable.

**Canniballs Object Roster**
| Object        | Starting Score | Color  | Shape    | Description                                                                                                                                               |
|---------------|----------------|--------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent         | 1              | Blue   | Circle   | RL agent for game. Can move one cell at a time in Manhattan distance (no diagonal) or stay still                                                                        |
| Food          | N/A            | Purple | Triangle | Food for agent to consume. If eaten, will randomly regenerate somewhere else on the map
| Random Walker | 1              | Green  | Circle   | Weakest Canniball opponent. Remains still with high probability. When active, moves randomly to adjacent cell, including diagonal.                        |
| Bouncer       | 2              | Yellow | Circle   | Most active Canniball. Bounces around according linearly or diagonally (changing directions when contacting other objects) with a very small probability of random motion and remains still roughly half the time.        |
| Chaser        | 3              | Red    | Circle   | Strongest Canniball. Remains still until agent comes within a certain radius. Then, it chases agent optimally with Manhattan distance, but with some probability of random motion |

**Example Screenshot and Additional Details**

<img align="left" src="images/canniballs_screenshot.png" alt="Canniballs Screenshot"  border=0 width=500 />

Canniballs is built to facilitate effective learning by the agent. A base negative reward is provided at each timestep to encourage movement in the agent. If food is consumed, a base reward is added to it. If the agent consumes an opponent, it receives a large reward proportional to the power score of the opponent. 

However, if it attempts to consume an opponent more powerful than itself, it incurs a large negative reward proportional to the difference in the agent's power score and the canniball opponents. This penalizes the agent less if it attempts to consume an opponent only slightly more powerful than itself. If the agent fails to consume an opponent, it itself is consumed and the episode terminates.

Lastly, if an object is ever trapped, it either remains still until it is freed or attempts a random motion to an arbitrary adjacent cell. In practice, this occurs rarely. The goal of this environment is to investigate the competency and certainty/interestingness of an agent in a simple environment that must conditionally prioritize its objectives. What can be consumed changes over time, leading to non-trivial strategy and tactics. In benchmark tests on a 2018 Macbook, the 22x22 environment can cycle at roughly 1200 FPS.

------------
<img src="images/canniballs_win.gif" alt="Canniballs Win GIF" align="right" border =0 width=390 />

To the right is a gif of a trained DQN agent in a small Canniballs environment achieving the goal with high precision. Note that although the agent does not completely maintain the correct semantic order of consumption (eating all weak canniballs first, then moving to next strongest) it generally does reflect the correct ordering. 

Moreover, though given no explicit reward signal for efficiency (agent is not penalized explicitly for taking its time) is given, the agent expresses a clear preference for canniballs in its immediate proximity and will temporarily abandon a chase if the agent is proving too difficult to catch.

Though simple, this demonstration also depicts the agents ability to quickly flip between "chase" and "flee", sometimes conducting both at the same time. For example, the red Canniball gives chase when the agent comes close looking for food. As it approaches, the agent zig-zags around to evade the Canniballs while also consuming weaker opponents and food. Afterwards, the agent then ceases evasion and attacks the read Canniball directly. Furthermore, all of this is completed in a very time efficient manner.


<a name = "known_issues"></a>
# Known Issues

**TLDR**: The best way to run a full pipeline is with `framework=tf2`, or tensorflow eager execution. However, depending on your model you may experience a memory leak, so plan checkpoints accordingly. Under the same conditions, there is no memory leak with a tf1 model, but tf1 rollout performance is 10x slower than tf2 agents. Moreover, we have not yet found an effective way to automatically port a TF1 checkpoint into TF2. We hope that RLlib-specific torch bugs will be fixed shortly and that torch can become the preferred framework for Cameleon execution.

### Training:
- Due to a [known bug](https://github.com/ray-project/ray/issues/16715) in the most recent RLlib release, training agents with `framework=torch` is broken. The bug is in triage and hopefully will be fixed soon. Cameleon creators have added a patch by which torch policies may be trained, but rollouts are still not possible and an area of current investigation.
- Training with tf2 and eager execution will cause a memory leak during training. The CPU RAM will progressively swell amongst all the workers during training and eventually terminate. This issue has been seen some before with RLlib, but there is no clear identification of the issue or when / if it will be fixed. A Cameleon-specific patch is also being explored.
- Although not a Cameleon issue per se, RLlib model-based algorithms do not support discrete action spaces and are therefore not yet supported with Cameleon.

### Rollouts:
- While it is possible to extract rollout information using `framework=tf1` (lazy execution) models, the implementation is currently inefficient and slow due to the overhead of repeatedly evaluating a large session graph. Fixes of this issue have not yet been successful under the current Cameleon design pattern. TF1 expert advice requested.
- RLlib and Gym monitors do not provide easy control of the artifacts they generate. Therefore, during rollout storage a cleanup script runs and clears out many of the unnecessary files. In the future it would be ideal to more directly control this behavior.

### General:
- Current the random seeding of the environment is flawed. Either all remote workers begin with the same seed and essentially replicate each other, or all begin with a completely random seed that is not tracked. Work to remedy this issue is ongoing.
- Currently environment wrappers are not idempotent and rely on specific orderings. An update to fix this is in progress.

<a name = "tips"></a>
# Usage and Debugging Tips

### Training:
- RLlib only supports two types of models out of the box for the core policy: MLPs and Convolutional models. In the latter case, the dimensionality of the environment and specific conv filter sizes / strides must be provided if the size differs from (84,84,k) or (42,42,k). With that said, many agents support LSTM, RNN, Attention, and other model augmentations of the core model with wrapping.

### Rollouts:
- Checkpoints will also read in the training config file. However, parts of this configuration will be overwritten if config kwargs are explicitly passed
- Rollouts can be run without a checkpoint installed, but general should provide a config that specifies details of model architecture
- RLlib is sensitive to filepaths, and does not provide intuitive errors. The first thing to check if experiencing an error is your checkpoint filepath
- Sometimes RLlib will inexplicably present a filepath error during rollouts, usually specifying a JSON file. Running the script again with no change usually resolves this.

### General Debugging and Tips:
- Always use the `manual_control.py` executable to validate game design and dynamics
- Benchmark the speed of aspects of your environment with `benchmark.py` before training
- If at all possible, do not run rollouts with a tf1 lazy evaluation agent. 

 
<a name = "support"></a>
# Support

If you have questions or issues with this package, you may post an issue or contact [Sam Showalter](mailto:sam.showalter@sri.com) or [Melinda Gervasio](mailto:melinda.gervasio@sri.com). 






