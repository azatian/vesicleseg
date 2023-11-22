# %%
import pandas as pd
dcvs = pd.read_csv("outputs/dcvsoma415.csv")
ratings = pd.read_csv("outputs/updated_merged_set_annotations_4_4_23.csv")
# %%
dcvs_cut = dcvs[["wk_id", "nucleus_id"]]
# %%
ratings_cut = ratings[["nucleus_id", "soma_dcv_rating"]]
# %%
ratings_cut_ddup = ratings_cut.drop_duplicates()
# %%
duplicates = ratings_cut_ddup["nucleus_id"].value_counts().reset_index()[:19].copy()
# %%
duplicates["consolidated"] = 0
# %%
#had to fix these manually 
duplicates.loc[0, "consolidated"] = 4
duplicates.loc[1, "consolidated"] = 2
duplicates.loc[2, "consolidated"] = 5
duplicates.loc[3, "consolidated"] = 4
duplicates.loc[4, "consolidated"] = 2
duplicates.loc[5, "consolidated"] = 2
duplicates.loc[6, "consolidated"] = 2
duplicates.loc[7, "consolidated"] = 1
duplicates.loc[8, "consolidated"] = 2
duplicates.loc[9, "consolidated"] = 5
duplicates.loc[10, "consolidated"] = 5
duplicates.loc[11, "consolidated"] = 3
duplicates.loc[12, "consolidated"] = 4
duplicates.loc[13, "consolidated"] = 4
duplicates.loc[14, "consolidated"] = 3
duplicates.loc[15, "consolidated"] = 3
duplicates.loc[16, "consolidated"] = 2
duplicates.loc[17, "consolidated"] = 1
duplicates.loc[18, "consolidated"] = 4



# %%
fixed = list(duplicates["index"])
# %%
set_one = ratings_cut_ddup[~ratings_cut_ddup["nucleus_id"].isin(fixed)]
# %%
set_two = duplicates[["index", "consolidated"]]
# %%
set_two = set_two.rename(columns={"index" : "nucleus_id", "consolidated" : "soma_dcv_rating"})
# %%
set_full = pd.concat([set_one, set_two])
# %%
set_full = set_full[~set_full["nucleus_id"].isnull()]
# %%
set_full["nucleus_id"] = set_full["nucleus_id"].astype(int)
# %%
merged = dcvs_cut.merge(set_full, how="left", left_on="nucleus_id", right_on="nucleus_id")
# %%
merged_df = merged[["wk_id", "soma_dcv_rating"]]
# %%
merged_df.to_csv("outputs/wk_id_to_rating_6_5_23.csv", index=False)
# %%
