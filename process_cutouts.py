# %% 
import pandas as pd
from datetime import datetime
from neurometry import viz, cave, ic
import matplotlib.pyplot as plt
import numpy as np
# %%
merged_set = pd.read_csv("data/merged_dcv_neuron_soma_sourcing_4_4_23.csv")

# %%
nucleus_ids = list(merged_set["nucleus_id"].unique().astype(int))
# %%
client = cave.set_up()
# %%
nucleus_tbl = cave.nucleus_query_by_id(client, nucleus_ids)
# %%
source = ic.source()
num = 1
for_df = []
id_to_location = {}
id_to_max_range = {}
id_to_root_id={}
for index, row in nucleus_tbl.iterrows():
    
    tupled = tuple(row["pt_position"])
    tupled = (int(tupled[0]//4), int(tupled[1]//4), int(tupled[2]//40))
    id_to_location[row["id"]] = tupled
    id_to_root_id[row["id"]] = row["pt_root_id"]
    bb_range = tuple([(row["bb_end_position"][0]-row["bb_start_position"][0])//4,
                (row["bb_end_position"][1]-row["bb_start_position"][1])//4,
                (row["bb_end_position"][2]-row["bb_start_position"][2])//40])
    max_bb_range = max(bb_range)

    id_to_max_range[row["id"]] = max_bb_range

    bounds = ic.create_bounds(id_to_location[row["id"]],max_bb_range+(max_bb_range//2), 6)
    imgvol, segdict = ic.get_img_and_seg(source, bounds, id_to_root_id[row["id"]])

    for_df.append(["dcvsoma" + str(num), row["id"], tupled, bb_range, max_bb_range,
                    row["pt_root_id"], row["pt_supervoxel_id"], row["volume"]])
    
    ic.writer("cutouts/dcvsoma" + str(num) + "/img/vol.tiff", np.transpose(imgvol))
    ic.writer("cutouts/dcvsoma" + str(num) + "/cellseg/cellseg.tiff", np.transpose(segdict[id_to_root_id[row["id"]]]*255).astype('uint8'))
    
    
    num += 1


# %% 
#had to code up this part because there was a break in the earlier code, internet connectivity issues?
continued = nucleus_tbl.loc[276:,:]
num = 277
for index, row in continued.iterrows():
    
    tupled = tuple(row["pt_position"])
    tupled = (int(tupled[0]//4), int(tupled[1]//4), int(tupled[2]//40))
    id_to_location[row["id"]] = tupled
    id_to_root_id[row["id"]] = row["pt_root_id"]
    bb_range = tuple([(row["bb_end_position"][0]-row["bb_start_position"][0])//4,
                (row["bb_end_position"][1]-row["bb_start_position"][1])//4,
                (row["bb_end_position"][2]-row["bb_start_position"][2])//40])
    max_bb_range = max(bb_range)

    id_to_max_range[row["id"]] = max_bb_range

    bounds = ic.create_bounds(id_to_location[row["id"]],max_bb_range+(max_bb_range//2), 6)
    imgvol, segdict = ic.get_img_and_seg(source, bounds, id_to_root_id[row["id"]])

    for_df.append(["dcvsoma" + str(num), row["id"], tupled, bb_range, max_bb_range,
                    row["pt_root_id"], row["pt_supervoxel_id"], row["volume"]])
    
    ic.writer("cutouts/dcvsoma" + str(num) + "/img/vol.tiff", np.transpose(imgvol))
    ic.writer("cutouts/dcvsoma" + str(num) + "/cellseg/cellseg.tiff", np.transpose(segdict[id_to_root_id[row["id"]]]*255).astype('uint8'))
    
    
    num += 1


# %%
dcv_df = pd.DataFrame(for_df, columns=["wk_id", "nucleus_id", "center", "bb_range", 
                                       "max_bb_range", "root_id", "supervoxel_id", "volume"])

dcv_df.to_csv("outputs/dcvsoma415.csv", index=False)
# %%
