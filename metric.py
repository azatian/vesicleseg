# %%
import pandas as pd
from sqlitedict import SqliteDict

# %%
df = pd.read_csv("data/codex_5000sample_9_18_23.csv")
database = SqliteDict('data/codex_5000sample_11_15_23.sqlite', autocommit=True)

# %%
temp = []
for key, item in database.items():
    temp.append([int(key), item[0], item[1]])


new_df = pd.DataFrame(temp, columns=["root_id", "vesicle_count", "volume_soma"])



# %%
merged = df.merge(new_df, how="inner", left_on="root_id", right_on="root_id")
# %%
merged_new = merged[["root_id", "vesicle_count", "volume_soma", "flow", "super_class",
                     "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "ito_lee_hemilineage",
                     "side", "nerve"]]
# %%
merged_new["saturation"] = (merged_new["vesicle_count"] / merged_new["volume_soma"]) * 1e5
# %%
merged_new = merged_new[["root_id", "saturation", "vesicle_count", "volume_soma", "flow", "super_class",
                     "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "ito_lee_hemilineage",
                     "side", "nerve"]]
# %%
merged_new_sort = merged_new.sort_values(by="saturation", ascending=False)
# %%
merged_new_sort.to_csv("data/codex5000_saturation_11_16_23.csv", index=False)
# %%
