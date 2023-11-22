# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import skimage.measure
import seaborn as sns
from neurometry import ml
# %%
with open('data/annotations_6_5_23.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    id_to_annotation = pickle.load(f)
# %%
df = pd.read_csv("data/wk_annotation_ids_6_3_2023.csv")
dff = df[df["do_not_use"] != 1]
id_to_vol = ml.ingestor(df)
collapsed_id_to_vol = ml.collapsor(id_to_vol)
# %%
sections = []
for key, value in collapsed_id_to_vol.items():
    temp = []
    _strings = key.split("_")
    temp.append(key)
    original = _strings[0]
    index = _strings[1]
    temp.append(original)
    temp.append(index)
    area =  (value > 0).sum()
    temp.append(area)
    labeled_image, count = skimage.measure.label(id_to_annotation[original][:,:,int(index)], return_num=True)
    temp.append(count)
    objects = skimage.measure.regionprops(labeled_image)
    object_areas = [obj["area"] for obj in objects]
    temp.append(object_areas)
    sections.append(temp)

cca_df = pd.DataFrame(sections, columns=["id", "original",
                                              "index", "area", "vesicle_count",
                                              "vesicle_sizes"])
# %%
cca_df_grp = cca_df.groupby("original").agg({"area" : "sum", "vesicle_count" : ["sum", "std"]}).reset_index()
# %%
cca_df_grp.columns = cca_df_grp.columns.to_flat_index()
lookup = {('original', ''): 'original', ('area', 'sum'): 'area_sum', 
          ('vesicle_count', 'sum') : 'vesicle_count_sum',
          ('vesicle_count', 'std') : 'vesicle_count_std'}
cca_df_grp_result = cca_df_grp.rename(columns=lookup)
# %%
sns.histplot(data=cca_df_grp_result, x="area_sum")
# %%
sns.histplot(data=cca_df_grp_result, x="vesicle_count_sum")
# %%
sns.histplot(data=cca_df_grp_result, x="vesicle_count_std")
# %%
cca_df_grp_result["vesicle_saturation"] = cca_df_grp_result["vesicle_count_sum"] / cca_df_grp_result["area_sum"]
cca_df_grp_result["vesicle_saturation"] = cca_df_grp_result["vesicle_saturation"] * 1e6
# %%
sns.histplot(data=cca_df_grp_result, x="vesicle_saturation")
# %%
vesicle_sizes_df = cca_df[["id", "original", "index", "vesicle_sizes"]].explode("vesicle_sizes").reset_index()
# %%
sns.histplot(data=vesicle_sizes_df, x="vesicle_sizes")
# %%
sns.histplot(data=vesicle_sizes_df.sort_values(by="vesicle_sizes")[:1500], x="vesicle_sizes")

# %%
id_to_annotation_cleaned = {}
for key, value in id_to_annotation.items():
    id_to_annotation_cleaned[key] = np.array(value)
    for i in range(6):
        labeled_image, count = skimage.measure.label(value[:,:,i], return_num=True)
        objects = skimage.measure.regionprops(labeled_image)
        #major filtering step
        small_objects =[obj for obj in objects if obj.area<30]
        for j in small_objects:
            id_to_annotation_cleaned[key][j.bbox[0]:j.bbox[2], j.bbox[1]:j.bbox[3], i]=0


# %%
sections = []
for key, value in collapsed_id_to_vol.items():
    temp = []
    _strings = key.split("_")
    temp.append(key)
    original = _strings[0]
    index = _strings[1]
    temp.append(original)
    temp.append(index)
    area =  (value > 0).sum()
    temp.append(area)
    labeled_image, count = skimage.measure.label(id_to_annotation_cleaned[original][:,:,int(index)], return_num=True)
    temp.append(count)
    objects = skimage.measure.regionprops(labeled_image)
    object_areas = [obj["area"] for obj in objects]
    temp.append(object_areas)
    sections.append(temp)

cca_df_cleaned = pd.DataFrame(sections, columns=["id", "original",
                                              "index", "area", "vesicle_count",
                                              "vesicle_sizes"])

cca_df_grp_cleaned = cca_df_cleaned.groupby("original").agg({"area" : "sum", "vesicle_count" : ["sum", "std"]}).reset_index()
cca_df_grp_cleaned.columns = cca_df_grp_cleaned.columns.to_flat_index()
lookup = {('original', ''): 'original', ('area', 'sum'): 'area_sum', 
          ('vesicle_count', 'sum') : 'vesicle_count_sum',
          ('vesicle_count', 'std') : 'vesicle_count_std'}
cca_df_grp_cleaned_result = cca_df_grp_cleaned.rename(columns=lookup)
cca_df_grp_cleaned_result["vesicle_saturation"] = cca_df_grp_cleaned_result["vesicle_count_sum"] / cca_df_grp_cleaned_result["area_sum"]
cca_df_grp_cleaned_result["vesicle_saturation"] = cca_df_grp_cleaned_result["vesicle_saturation"] * 1e6
vesicle_sizes_df_cleaned = cca_df_cleaned[["id", "original", "index", "vesicle_sizes"]].explode("vesicle_sizes").reset_index()
# %%
sns.histplot(data=vesicle_sizes_df_cleaned, x="vesicle_sizes")
# %%
sns.histplot(data=vesicle_sizes_df_cleaned.sort_values(by="vesicle_sizes")[:1500], x="vesicle_sizes")
# %%
#good example to show
plt.imshow(id_to_annotation["dcvsoma317"][:,:,2], cmap=plt.cm.gray)
# %%
plt.imshow(id_to_annotation_cleaned["dcvsoma317"][:,:,2], cmap=plt.cm.gray)

# %%
#uncommet if needed
#with open('data/annotations_cleaned_6_5_23.pickle', 'wb') as handle:
#    pickle.dump(id_to_annotation_cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
#saving cleaned connected component analysis
#cca_df_cleaned.to_csv("outputs/cca/cca_cleaned_6_5_23.csv", index=False)
#cca_df_grp_cleaned_result.to_csv("outputs/cca/cca_cleaned_grouped_6_5_23.csv", index=False)
#vesicle_sizes_df_cleaned[["id", "original", "index", "vesicle_sizes"]].to_csv("outputs/cca/cca_vesicle_sizes_cleaned_6_5_23.csv", index=False)


# %%
#post analysis to plot rough vesicle saturation
master = pd.read_csv("outputs/wk_id_to_rating_6_5_23.csv")
# %%
merged = master.merge(cca_df_grp_cleaned_result, how="inner", left_on="wk_id", right_on="original")
# %%
sns.violinplot(data=merged, x="soma_dcv_rating", y="vesicle_saturation", cut=0, scale="count")
# %%
