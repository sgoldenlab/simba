import numpy as np
import pandas as pd

from simba.mixins.plotting_mixin import PlottingMixin

GI_PATH = r"C:\troubleshooting\mitra\project_folder\logs\straub_tail\straub_tail_aggregates\straub_tail_aggregates_gi.csv"
GQ_PATH = r"D:\troubleshooting\mitra\project_folder\logs\straub_tail_data\aggregate_straub_tail\straub_tail_aggregates.csv"

gi_df = pd.read_csv(GI_PATH)
gq_df = pd.read_csv(GQ_PATH)

gi_df.columns = [x.lower() for x in gi_df.columns]
gq_df.columns = [x.lower() for x in gq_df.columns]



gi_df['video'] = gi_df['video'].str.lower()
gq_df['video'] = gq_df['video'].str.lower()

conditions_1 = [
    gi_df['video'].str.contains('_gi_cno_'),
    gi_df['video'].str.contains('_gi_saline_'),
    gi_df['video'].str.contains('_gq_cno_'),
    gi_df['video'].str.contains('_gq_saline_'),
]

conditions_2 = [
    gq_df['video'].str.contains('_cno'),
    gq_df['video'].str.contains('_saline'),
]


choices_1 = ['Gi CNO', 'Gi Saline', 'Gq CNO', 'Gq Saline']
choices_2 = ['Gq CNO', 'Gq Saline']

gi_df['group'] = np.select(conditions_1, choices_1, default='Unknown')
gq_df['group'] = np.select(conditions_2, choices_2, default='Unknown')
gq_df['experiment'] = 2
gi_df['experiment'] = 1

PlottingMixin.plot_bar_chart(df=gi_df, x='group', y='straub_tail - total event duration (s)')

