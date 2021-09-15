#################################################################################
#
#             Project Title:  Visuals for Analysis
#             Author:         Sam Showalter
#             Date:           2021-09-15
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import copy
import shutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#################################################################################
#   Function-Class Declaration
#################################################################################

df = pd.read_csv('data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Hard-12x12-v0_ep329_rs42_w10/1-interaction/execution-uncertainty/mean-exec-div-time.csv')

print(df.columns)

sdf = df[['agent_pos_x','agent_pos_y','mean_action_execution_div']].values.tolist()

# print(sdf)

support = np.zeros((12,12))
uncert = np.zeros((12,12))
for i in range(len(sdf)):
    x,y,u = sdf[i]
    x,y = int(x),int(y)
    support[x,y] +=1
    uncert[x,y] += u

plot = sns.heatmap(uncert/support)
plt.savefig('test_heat.png')






#################################################################################
#   Main Method
#################################################################################



