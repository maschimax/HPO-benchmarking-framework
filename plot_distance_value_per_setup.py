import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = 'turbofan'
setup_variants = [(1, False), (8, False), (1, True)]
ipt_colors = ['#179c7d', '#438cd4', '#ffcc99']
bar_width = 0.35

file_path = './hpo_framework/results/%s/RankingAnalysis/%s_ranked_metrics.csv' % (
    dataset, dataset)

ranked_df = pd.read_csv(file_path, index_col=0)
hpo_techs = ranked_df['1st metric HPO-technique'].unique()

fig, ax = plt.subplots(figsize=(9, 5))

dv_per_setup_df = pd.DataFrame([])

for i in range(len(setup_variants)):

    this_setup = setup_variants[i]

    setup_df = ranked_df.loc[(ranked_df['Workers'] == this_setup[0]) &
                             (ranked_df['Warm start'] == this_setup[1]) & (
                             (ranked_df['1st metric'] == 'Max Cut Validation Loss') |
                             (ranked_df['1st metric'] == '2nd Cut Validation Loss') |
                             (ranked_df['1st metric'] == '3rd Cut Validation Loss') |
                             (ranked_df['1st metric'] == '4th Cut Validation Loss')), :]

    for this_tech in hpo_techs:

        this_distance_value = np.nanmean(setup_df.loc[(setup_df['1st metric HPO-technique'] == this_tech),
                                                      '1st metric scaled deviation'].to_numpy())

        dv_per_setup_df.loc[i, this_tech] = this_distance_value

    if i == 0:
        x_pos = np.array(range(len(hpo_techs))) - bar_width
    elif i == 1:
        x_pos = np.array(range(len(hpo_techs)))
    elif i == 2:
        x_pos = np.array(range(len(hpo_techs))) - bar_width
    else:
        raise Exception('Too many setup variants!')

    ax.bar(x=x_pos, height=dv_per_setup_df.loc[i, :].to_numpy(), color=ipt_colors[i])

bla = 0