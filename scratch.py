"""
TODO:

    - Documentation for Canniballs on README
    - Automate the training API
    - Build wrappers for CAMeLeon Env
    - Tidy up registration.py
    -

"""

# from cameleon.utils.env_utils import str2wrapper
import json
a="""{
"num_gpus":1,
"num_workers":10,
"model":{"dim":12,
         "max_seq_len":0,
         "conv_filters":[[16,[3,3],1],
                         [32,[3,3],2],
                         [512,[6,6],1]]
        }
}"""
# print(json.loads(a))

# print(str2wrapper("encoding_only canniballs_one_hot"))

import ray
from ray import tune
import numpy  as np

a = np.array([1,2,3])

# print(a is not None)
# if __name__ == "__main__":
#     ray.init()
#     tune.run("PPO",
#              config={"env": "CartPole-v0", "framework": "torch", "num_gpus": 1,
#                      "num_workers": 10,},
#              # resources_per_trial={"cpu":1,"gpu":1},
#              local_dir = "rollouts/tune/",
#              verbose = 1,
#              checkpoint_freq = 1,
#              # name = PPO
#              # trial_name_creator=lambda trial: "test_ppo_trial_info",
#              trial_dirname_creator=lambda trial: "test_dir",
#              stop = {"training_iteration":1000})

# from ray.tune.trial
import datetime as dt

# time_status = "Trial {:2d} | eta {} | {:6.2f}% complete |avg_per_iter {:6.2f}".format(
#                         10,
#                         str(dt.timedelta(seconds = round(10934))),
#                         10*100/1023,
#                         45.2)
import re
a = re.match(r'(\d+)_ep(\d+)_s(\d+)_r*(.+).[ph]kl',"1010_ep12_s1010_rn3.hkl")
print(a.groups())
print("HI")
