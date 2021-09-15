import sys
a = """
TODO:

    <--- High Priority --->
    - Think of new / better disruptions
    - Read and notes on ICML articles for OOD to help with imagination

    <--- Done --->
    X Create hash to easily look up rollouts
    X Work with Pedro to better integrate Cameleon and ixdrl (especially with artifact saving)
    X Update Pedros ppt figure for presentation
    X Cite Pedros paper in presentation
    X Align interestingness artifacts so easy to line up rollouts (not so easy right now)
    X Add starting iterations so the progressbar makes sense ( if you start from a checkpoint)
    X FIX huge bug with rollout misalignment of observations, actions and rewards. Line it up with video.
    X Figure out if or how to make wrappers more idempotent (NOTE: Decided to just document order, idempotence a bad idea)
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
    X Dynamic saving onto filex from archive
    X Convert dynamic saving into a CLI
    X Add transferrence of artifacts information to README
    X Spell check on README docs
    X Add random seed to training
    X Add random seed to model name
    X Update readme with random seed information

    -- Week of Aug 17 --
    X Add policy extractors to other actor critic architectures
    X Start outlining experiments for competency
    X Begin training models for Actor-critic competency faceoff
    X Standardize email bot functionality Cameleon
    X Keep training models for competency faceoff
    X Train automation experiment scripts
    X Rollout automation experiment scripts
    X Reach out to Daniel to begin strategy integration
    X Add info for Pedro iRad
    X Test and finalize interestingness experiment executable
    X Push metadata store idea through cameleon pipeline
    X Change all print statements to log statements

    -- Week of Aug 23 --
    X Add way to know what env model was trained on to rollouts (argparse JSON dict)
    X Retrain DQN agent with fewer checkpoints
    X Add metadata files to all rollout directories
    X Training investigate random seeding
    X Alter interestingness code so it lines up with cameleon
       X Bundle interestingness by rollout folder
       X Separate .csvs and plots for each rollout
       X Name interestingness files in an easy way
       X Identify suitable naming convention for dir nest
    X Model name update on progress update for training
    X Ray shutdown after training iterations to prevent seeding issues
    X Save metadata earlier for training and rollouts (can always resave)
    X Interestingness modules
       x reward
       x value
       x action value
    X Add way to convert encoded environment into rgb
    X Clean up iXDRL code (leaving most of it to pedro)
    X Improve modularity of way that environment is encoded for interestingness
    X Post interestingness code to filex
    X Clean up cameleon code
    X Update documentation to reflect new cameleon execution format

    -- Week of Aug 31 --
    X Generate insightful plots from code output (interestingness)
    X Update interestingness to accommodate multiple checkpoints of same model
    X Update cameleon to facilitate more intuitive naming for checkpoints from same model
    X Add content for AIC seminar for interestingness
    X Post new models to filex

    -- Week of Sept 7 --
    X Add disruptions to CAMeLeon README
    X Update code to be more modular around disruptions
    X Add way for mailbot to send error messages effectively and gracefully
    X Change way log level is set
    X Final pass-through on CAMeLeon code. Update any bad design patterns.
    X Update README again, hopefully for the last time
    X Update way environment ingests info about model
    X Update examples code and finalize documentation




"""




#######################################################################


# p1 = "data/interestingness/IMPALA_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-12x12-v0_ep147_rs42_w5/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# p2 = "data/interestingness/IMPALA_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Medium-12x12-v0_ep160_rs42_w5/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# p3 = "data/interestingness/IMPALA_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Hard-12x12-v0_ep141_rs42_w5/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# # exec_cert_path = "data/interestingness/IMPALA_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Hard-12x12-v0_ep141_rs42_w5/1-interaction/value/value-time.csv"

# paths = [p1,p2,p3]
# levels = ['easy','medium','hard']


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# total_df = None

# for p,l in zip(paths,levels):
#     print(l)
#     print(p)
#     df = pd.read_csv(p)
#     print(df.shape)
#     df = df[df.Timestep < 150]
#     df['level'] = l

#     if total_df is None:
#         total_df = df
#         print(df.shape)

#     else:
#         total_df = pd.concat([total_df,df],axis = 0)
#         print(df.shape)




# plot = sns.lineplot(data = total_df, x = "Timestep", y="mean_action_execution_div",hue = "level")

# plt.savefig("mean_value_IMPALA_all_levels.png")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Medium-12x12-v0_ep1204_rs42_w10/1-interaction/execution-value/mean-exec-value-diff-time.csv"

# Execution certainty
# path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0_ep187_rs42_w1/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# path1 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0_ep187_rs42_w1/1-interaction/execution-uncertainty/mean-exec-div-time.csv"

# # Execution certainty hard
# path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0_ep1784_rs42_w10/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# path1 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0_ep1781_rs42_w10/1-interaction/execution-uncertainty/mean-exec-div-time.csv"


# Value
path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0_ep1275_rs42_w10/1-interaction/value/value-time.csv"
path1 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0_ep1781_rs42_w10/1-interaction/value/value-time.csv"


# # Execution certainty
# path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/chaser_Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0_ep1784_rs42_w10/1-interaction/execution-uncertainty/mean-exec-div-time.csv"
# path1 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0_ep1781_rs42_w10/1-interaction/execution-uncertainty/mean-exec-div-time.csv"




# # Value hard
# path2 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Hard-Corner-Disruption-12x12-v0_ep178_rs42_w1/1-interaction/value/value-time.csv"
# path1 = "data/interestingness/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0/Cameleon-Canniballs-Hard-Fake-Corner-Disruption-12x12-v0_ep168_rs42_w1/1-interaction/value/value-time.csv"

# df1 = pd.read_csv(path1)
# df2 = pd.read_csv(path2)
# df1['type'] = 'no_disruption'
# df2['type'] = 'disruption'

# df3 = pd.read_csv(path3)
# # df3 = pd.read_csv(path3)

# # cnt_df = df[['timestep','model_checkpoint']].groupby(['timestep']).agg('count')
# # plt.figure(figsize=(16,10))
# dfs = [df1,df2]


# # for i in range(3):
# #     dfs[i] = dfs[i][['model_checkpoint','episode','timestep']]
# df = pd.concat(dfs,axis =0)


# # df['action_diff'] = df['action_max_min_diffs'].str.replace('[','').str.replace("]","").astype(float)
# # df['reward'] = df.episode.apply(lambda x: int(x.split("_")[-1].replace("r","").replace("n","-")))

# # print(df.reward)
# # df['reward']= df.reward.apply(lambda x: x[-1]("n","-").astype(int))

# df = df[df.timestep < 51]
# df.drop('timestep',inplace = True,axis = 1)
# final_df = final_df[~final_df.model_checkpoint.isin([20,40,400,1400,600,800,1800,1600,1200,420,440,460,200,480])]
# # print(df.model_checkpoint.drop_duplicates())

# # cnt_df.to_csv('count_df.csv',index = False)
# # final_df = final_df[~df.model_checkpoint.isin([12
# palette = sns.color_palette("mako_r", len(final_df.model_checkpoint.drop_duplicates()))
# palette1 = sns.color_palette("mako_r", 2)
# palette2 = sns.color_palette("Accent", 2)

# # box_df = df[['reward','model_checkpoint','episode']].groupby("episode").agg({"reward":['sum']})
# # box_df = df[['reward','model_checkpoint','episode']]
# # print(box_df)
# # sys.exit(1)
# # sns.boxplot(data=box_df, x = 'model_checkpoint',y = 'reward')
# print(df.columns)

# # Execution certainty
# plot = sns.lineplot(data = df, x = "timestep", y="mean_action_execution_div",hue = "type",legend = "full",palette=palette)
# plt.legend(bbox_to_anchor=(0.5, 1.10), ncol = 2, fancybox=True, loc = 'upper center')
# plt.tight_layout()
# plt.savefig("APPO_exec-cert_easy_disruption_chaser.png")


# # Value
# sns.lineplot(data = df, x = "timestep", y="value",hue = "type",legend = "full",palette=palette1)
# sns.lineplot(data = df, x = "timestep", y="actual_value_to_go",hue = "type",dashes = True, palette = palette2)
# plt.legend(bbox_to_anchor=(0.5, 1.10), ncol = 4, fancybox=True, loc = 'upper center')
# plt.tight_layout()
# plt.savefig("APPO_exec-value_easy_disruption_food_vtg.png")

#######################################################################
# Random test
#######################################################################

import numpy as np

a =[1,2,3,4]
np.random.shuffle(a)
print(a)


