import pandas as pd

# --- Read the original data to check its structure ---
data = pd.read_csv(r'E:\dlc_pose_annotations\Tvideo_0\CollectedData_SN.csv')
print(f"Original data shape: {data.shape}")
print(f"Original columns: {len(data.columns)}")

# Process the data
data = data.loc[2:, ].reset_index(drop=True)
data_split = data['scorer'].str.split(r'\\', expand=True)
df_new = pd.concat([data_split, data], axis=1).drop('scorer', axis=1)

print(f"df_new shape: {df_new.shape}")
print(f"df_new columns: {len(df_new.columns)}")

# Now create columns MultiIndex that matches df_new column count
# df_new has 3 index columns + original data columns
num_data_cols = len(df_new.columns) - 3  # subtract the 3 split columns

scorer = ["", "", ""] + ["SN"] * num_data_cols
individuals = ["", "", ""] + [1]*(num_data_cols//2) + [2]*(num_data_cols//2)
bodyparts = ["", "", ""] + [
    "Ear_left","Ear_left","Ear_right","Ear_right",
    "Nose","Nose","Center","Center",
    "Lateral_left","Lateral_left","Lateral_right","Lateral_right",
    "Tail_base","Tail_base","Tail_end","Tail_end"
] * 2
coords = ["", "", ""] + ["x","y"] * (num_data_cols//2)

columns = pd.MultiIndex.from_arrays(
    [scorer, individuals, bodyparts, coords],
    names=["scorer", "individuals", "bodyparts", "coords"]
)

print(f"MultiIndex columns: {len(columns)}")

# Create DataFrame with matching columns
df = pd.DataFrame(columns=columns, data=df_new.values)

# Set the index
df_new.set_index([0, 1, 2], inplace=True)
df.index = df_new.index

print("\nFinal DataFrame:")
print(df.head())
print("\nShape:", df.shape)

