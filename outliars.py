# %%
import pandas as pd


# %%
df = pd.read_csv("data/wk_annotation_ids_6_3_2023.csv")
#def main() -> None:
#    print("-----------STARTING PIPELINE---------------------------")
    
# %%
dff = df[df["do_not_use"] != 1]
# %%
from neurometry import ml
# %%
id_to_vol = ml.ingestor(df)
# %%
import matplotlib.pyplot as plt
import numpy as np
#no need to transpose or rotate 
#plt.imshow(id_to_vol["dcvsoma110"][:,:,0], cmap=plt.cm.gray)
# %%
collapsed_id_to_vol = ml.collapsor(id_to_vol)
# %%
pixel_sum = []
for key, value in collapsed_id_to_vol.items():
    temp = []
    _strings = key.split("_")
    temp.append(key)
    temp.append(_strings[0])
    temp.append(_strings[1])
    temp.append(value.shape[0])
    temp.append(value.shape[1])
    temp.append(value.sum())
    temp.append(value.mean())
    temp.append(value.std())
    area =  (value > 0).sum() / (value.shape[0] * value.shape[1])
    temp.append(area)
    pixel_sum.append(temp)

#normalized_mutual_information_analysis
# %%
metrics_df = pd.DataFrame(pixel_sum, columns=["id", "original",
                                              "index", "length", "width",
                                              "sum", "mean", "std", "area_percentage"])
# %%
incluster_metrics = metrics_df.groupby("original").agg({"mean" : "std", "std" : "std", "area_percentage" : "std"}).reset_index()
# %%
incluster_metrics["sum_of_stds"] = incluster_metrics["mean"] + incluster_metrics["area_percentage"]
# %%
import seaborn as sns
# %%
sns.histplot(data=metrics_df, x="sum")
# %%
sns.histplot(data=metrics_df, x="mean")
# %%
sns.histplot(data=metrics_df, x="std")
# %%
sns.histplot(data=metrics_df, x="area_percentage")
# %%
sns.histplot(incluster_metrics, x="mean")
# %%
sns.histplot(incluster_metrics, x="area_percentage")
# %%
sns.histplot(incluster_metrics, x="std")
# %%
sns.histplot(incluster_metrics, x="sum_of_stds")
# %%
