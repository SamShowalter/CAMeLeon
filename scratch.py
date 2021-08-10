
"""
TODO:

    <--- Done --->
    X Create hash to easily look up rollouts
    X Update Pedros ppt figure for presentation
    X Cite Pedros paper in presentation
    X Align interestingness artifacts so easy to line up rollouts (not so easy right now)
    X Add starting iterations so the progressbar makes sense ( if you start from a checkpoint)
    X FIX huge bug with rollout misalignment of observations, actions and rewards. Line it up with video.
    X Figure out if or how to make wrappers more idempotent (NOTE: Decided to just document order)
    X Update rollout code so that random agent can be used to make rollouts (+ document it)
    X Update training so that random agent is saved for later review with competency
    X Fix bug in interestingness for rollout data
    X Fix rollouts with standalone model
    X Add different training durations
    X Adjust naming for rollouts for timesteps
    X Add more hierarchical saving for interestingness to prevent clashes
    X Update readme with documentation changes
    X Update rollout documentation
    X Clean rollout code
    X Clean training code
    X Read 2 ICML articles and take notes
        x Update team with any valuable information
    X Allow training by number of episodes or timesteps (maybe best to just use tune)
    X Update Cameleon example code
    X Make sure required arguments are specified correctly

    <--- High Priority --->
    - Spell check on README docs
    - Start outlining experiments for competency
    - Work with Pedro to better integrate Cameleon and ixdrl (especially with artifact saving)

    <--- Low priority --->
    - Make progress on gvf artifacts

"""

import hashlib
import numpy as np

# Use this for rollouts!
data = np.random.rand(12,12,4)
# print(data.shape)
# h = hashlib.shake_256(str(data).encode()).hexdigest(4)
# f = hashlib.shake_256(str(np.array([1,2,3,4,5,7,6])).encode()).hexdigest(4)

# print(f)
# print(h)

# a = r'_pid(\d+)-(\d+).[ph]kl'

# import re
import re
path ="rollouts/DQN_torch_Cameleon-Canniballs-Medium-12x12-v0_ep10_rs42_w5_2021.08.09/"
match = re.findall(r'_rs(\d+)_w(\d+)',path)[0]
print(match.components())
