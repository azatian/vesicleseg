# %%
from sqlitedict import SqliteDict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#from multiprocessing import Pool
db = SqliteDict("data/root_to_mask_9_18_23.sqlite", encode="ascii")

# %%
for key, item in db.items():
    print("%s=%s" % (key, item))
    break
# %%
int.from_bytes(key, byteorder='little')
# %%
#convert bytes to int
def bytes_to_int(b):
    return int.from_bytes(b, byteorder="little")

#convert int to bytes
def int_to_bytes(i):
    return i.to_bytes((i.bit_length() + 7) // 8, byteorder='little')

# %%
#calculates what percentage of the cutout is filled
def signal(slice, x, y):
    total = np.count_nonzero(slice)
    return (total / (x*y)) * 100

#
#percentages = []
#cutout = db[int_to_bytes(720575940629248374)]
#for i in range(z):
#    percentages.append(signal(cutout[:,:,i], 732, 732))
#df = pd.Series(percentages).reset_index()
# %%
#plots histogram of percentage fills 
def lineplot(df):
    return sns.lineplot(x="index", y=0, data=df).set(xlabel='section index', ylabel='percentage nonzero')

def lineplot_rename(df):
    return sns.lineplot(x="index", y="filled", data=df).set(xlabel='section index', ylabel='percentage nonzero')
# %%
#input: root id
#output: dataframe of slice index and percentage filled

def process_root(root):
    percentages = []
    try:
        cutout = db[int_to_bytes(root)]
    except:
        return "key not found"
    x = cutout.shape[0]
    y = cutout.shape[1]
    z = cutout.shape[2]
    for i in range(z):
        percentages.append(signal(cutout[:,:,i], x, y))
    df = pd.Series(percentages).reset_index()
    return df

# %%
new_ids_df = pd.read_csv("data/codex_5000sample_9_18_23.csv")
new_ids = list(new_ids_df["new_id"])

# %%
#this was was for 45 minutes on a single pool, consider multiprocessi9ng

not_found = []
found = []
for id in new_ids:
    df = process_root(id)
    if type(df) == str:
        not_found.append(df)
    else:
        df["root"] = id
        found.append(df)


# %%
grouped = pd.concat(found)

# %%
grouped.rename(columns={0: "filled"}, inplace=True)

# %%
grouped.to_csv("data/codex_5000sample_slicekey_9_18_23.csv", index=False)
#sns.histplot(data=grouped, x="filled")
#plt.imshow(root_to_mask[720575940614448059][:,:,40], cmap=plt.cm.gray)
#grouped.sort_values(by="filled")[:8000].groupby(["root"])["root"].agg(["count"]).reset_index().sort_values(by="count", ascending=False)[:50]
#mean plot, some issues to resolve
#sns.histplot(data=grouped.groupby(["root"])["filled"].agg(['mean']).reset_index(), x="mean")
# %%
db.close()


#How to crop an image?
# Mask of non-black pixels (assuming image has a single channel).
#mask = image > 0

# Coordinates of non-black pixels.
#coords = np.argwhere(mask)

# Bounding box of non-black pixels.
#x0, y0 = coords.min(axis=0)
#x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

# Get the contents of the bounding box.
#cropped = image[x0:x1, y0:y1]