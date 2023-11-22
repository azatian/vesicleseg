#This module is responsible for the following
#Input: A Root ID
#     Additional Info about Root ID: identity and connectivity
#Output: Vesicle Saturation Metric
#Procedure: Use nuclei segmentation of root id to estimate bounding box 
#encompassing soma, create sections, filter, apply working model to each section,
#count up segmented bodies, divide by total area seen
# %%
import random
from fafbseg import flywire
import matplotlib.pyplot as plt
import numpy as np
# %%
from neurometry import viz, cave, ic
import pandas as pd
from sqlitedict import SqliteDict
client = cave.set_up()

# %%
meta = cave.get_cellinfo_meta(client)
# %%
codex = pd.read_csv("data/annotation_export_for_codex_310323.csv")
# %%
unique_ids = list(codex["root_id"].unique())
# %%
#batch_1 = random.sample(unique_ids, 200)
#batch_1 = random.sample(unique_ids, 5000)
# %%
#flywire.is_latest_root(batch_1)
# %%
#batch_1_update = flywire.update_ids(batch_1)
# %%
#merged = codex.merge(batch_1_update, how="inner", left_on="root_id", right_on="old_id")
# %%
#new_ids = list(merged["new_id"])
new_ids_df = pd.read_csv("data/codex_5000sample_9_18_23.csv")
new_ids = list(new_ids_df["new_id"])
# %%
#Nucleus coordinates div ide by 4, 4, 40?
#Encompass z +/- 10 sections
#if there are two or more nuclei IDs, take the one with the bigger volume

nuclei_tbl = client.materialize.query_table('nuclei_v1', filter_in_dict={"pt_root_id" : new_ids})
# %%
#my_example = nuclei_tbl[nuclei_tbl["pt_root_id"] == 720575940604341280]
# %%
source = ic.source()
root_to_location = {}
root_to_max_range = {}
#root_to_vol = {}
#root_to_seg = {}
root_to_mask = SqliteDict('data/root_to_mask_9_18_23.sqlite', autocommit=True)
#root_to_mask = {}
counter = 0
for x in list(nuclei_tbl["pt_root_id"].unique()):
    ex = nuclei_tbl[nuclei_tbl["pt_root_id"] == x]
    if len(ex) > 1:
        ex = ex[ex["volume"] == max(ex["volume"])]
        if len(ex) > 1:
            ex = ex.sample(1)
    
    ex = ex.reset_index()
    tupled = tuple(ex["pt_position"])[0]
    tupled = (int(tupled[0]//4), int(tupled[1]//4), int(tupled[2]//40))
    root_to_location[x] = tupled

    bb_range = tuple([(ex["bb_end_position"][0][0]-ex["bb_start_position"][0][0])//4,
                (ex["bb_end_position"][0][1]-ex["bb_start_position"][0][1])//4])
    z_range = (ex["bb_end_position"][0][2]-ex["bb_start_position"][0][2]+20)//40
    max_bb_range = max(bb_range)
    root_to_max_range[x] = max_bb_range
    bounds = ic.create_bounds(root_to_location[x],max_bb_range+(max_bb_range//2), z_range)
    imgvol, segdict = ic.get_img_and_seg(source, bounds, x)
    #root_to_vol[x] = imgvol
    #root_to_seg[x] = segdict

    raw = np.array(imgvol*(segdict[x]), dtype=np.uint8)
    #counterclockwise rotation by 90 degrees
    rotated = np.rot90(raw).copy()
    #flip vertical
    flipped = np.flipud(rotated).copy()

    root_to_mask[x] = flipped
    counter += 1
    print(counter)


#pickle your findings to continue working on this tomorrow
# %%
'''
import pickle
with open('data/root_to_vol_8_16_23.pickle', 'wb') as handle:
    pickle.dump(root_to_vol, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open('data/root_to_seg_8_16_23.pickle', 'wb') as handle:
    pickle.dump(root_to_seg, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
import pickle
with open('data/root_to_vol_8_16_23.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    root_to_vol = pickle.load(f)

with open('data/root_to_seg_8_16_23.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    root_to_seg = pickle.load(f)



#for index, row in nuclei_tbl.iterrows():
'''
# %%
#import matplotlib.pyplot as plt
# %%
#root_to_mask = {}
#for key, value in root_to_vol.items():
#    raw = np.array(value*(root_to_seg[key][key]), dtype=np.uint8)
#    rotated = np.rot90(raw).copy()
    #flip vertical
#    flipped = np.flipud(rotated).copy()
#
#    root_to_mask[key] = flipped

#import pickle
#with open('data/root_to_mask_9_18_23.pickle', 'wb') as handle:
#    pickle.dump(root_to_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
#plt.imshow(root_to_mask[720575940614448059][:,:,40], cmap=plt.cm.gray)
#import pickle
#with open('data/root_to_mask_9_18_23.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
#    root_to_mask = pickle.load(f)
# %%
root_to_mask.close()
