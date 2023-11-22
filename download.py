# %%
import webknossos as wk
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imread
import pandas as pd
import pickle
# %%
ids = pd.read_csv("data/wk_annotation_ids_6_3_2023.csv")
ids = ids[ids["do_not_use"] != 1]
ds_path = "/home/tmn7/wkw-test/wkw2/binaryData/harvard-htem/"
# %%
id_to_annotation = {}
not_working = []
MAG = wk.Mag("1")
for index, row in ids.iterrows():
     #print(row["annotation_id"])
     ds = wk.dataset.Dataset('annotations/tmp', voxel_size=(4,4,40))
     img = wk.Dataset.open(ds_path + row["wk_id"])
     bbox = img.get_layer("img").bounding_box
     try:
        with wk.webknossos_context(
                token="9ep0OkwPk41MnnKzqNGh9w",
                url="http://catmaid2.hms.harvard.edu:9000"):
            annotation = wk.Annotation.download(row["annotation_id"])
            annotation.export_volume_layer_to_dataset(ds)
        
        ds.get_segmentation_layers()[0].bounding_box = bbox
        mag_view = ds.get_segmentation_layers()[0].get_mag(MAG)
        data = mag_view.read()[0,:,:,:]
        id_to_annotation[row["wk_id"]] = data
        ds.delete_layer('volume_layer')
     except:
         not_working.append(row["wk_id"])

# %%
from neurometry import ml
id_to_vol = ml.ingestor(ids)

# %%
#transformer operation
import numpy as np
for key, value in id_to_annotation.items():
    id_to_annotation[key] = np.flipud(np.rot90(value))

#to fill in blank annotations
for x in not_working:
    id_to_annotation[x] = np.zeros(id_to_vol[x].shape)
#plt.imshow(id_to_vol["dcvsoma110"][:,:,0], cmap=plt.cm.gray)
#vol = imread("cutouts/dcvsoma323/img/vol.tiff")
#seg = imread("cutouts/dcvsoma323/cellseg/cellseg.tiff")
#vol = np.array(vol).T
#seg = np.array(seg).T
#plt.imshow(np.flipud(np.rot90(vol[:,:,0])), cmap=plt.cm.gray)
# %%
shape_checker = []
for key, value in id_to_annotation.items():
    shape_checker.append([key, id_to_vol[key].shape, value.shape, 
                          id_to_vol[key].shape == value.shape])

# %%
shape_checker_df = pd.DataFrame(shape_checker, columns=["wk_id", "vol_shape", "annotation_shape", "truth_value"])

# %%
with open('data/annotations_6_5_23.pickle', 'wb') as handle:
    pickle.dump(id_to_annotation, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
#import skimage.measure
# labeled_image, count = skimage.measure.label(id_to_annotation["dcvsoma110"][:,:,0], return_num=True)
# objects = skimage.measure.regionprops(labeled_image)
# object_areas = [obj["area"] for obj in objects]
# %% 

#did not run code below after here, just testing stuff
# %%
with wk.webknossos_context(
    token="9ep0OkwPk41MnnKzqNGh9w",
    url="http://catmaid2.hms.harvard.edu:9000"
):
    annotation = wk.Annotation.download(
        "642ef10f0100009a00afdc17",
        annotation_type="Explorational",
    )
# %%
with wk.webknossos_context(
    token="9ep0OkwPk41MnnKzqNGh9w",
    url="http://catmaid2.hms.harvard.edu:9000"
):
    dataset = wk.Annotation.open_as_remote_dataset(
        "642ef10f0100009a00afdc17"
    )
# %%
dataset = wk.Annotation.open_as_remote_dataset(
        "642ef10f0100009a00afdc17", webknossos_url="https://webknossos.org"
    )
# %%
dataset = annotation.open_as_remote_dataset("642ef10f0100009a00afdc17")
# %%

# %%
dataset = wk.Dataset.open("annotations/dcvsoma323_annotation/")
# %%
import wkw
# %%
dataset3 = wkw.Dataset.open('annotations/dcvsoma323_annotation/1')
# %%
dataset2 = wk.Dataset.open_remote("http://catmaid2.hms.harvard.edu:9000/data/annotations/zarr/4pJXHDyVK_rvi7db/")
# %%
# %%
with wk.webknossos_context(
    #token="9ep0OkwPk41MnnKzqNGh9w",
    url="http://catmaid2.hms.harvard.edu:9000"
):
    dataset = wk.Annotation.open_as_remote_dataset(
        "4pJXHDyVK_rvi7db"
    )
# %%
with wk.webknossos_context(
    token="9ep0OkwPk41MnnKzqNGh9w",
    url="http://catmaid2.hms.harvard.edu:9000"
):
    dataset = wk.Dataset.open_remote(
        "http://catmaid2.hms.harvard.edu:9000/data/annotations/zarr/4pJXHDyVK_rvi7db/"
    )
# %%
dataset2 = wk.Dataset.open("http://catmaid2.hms.harvard.edu:9000/data/annotations/zarr/4pJXHDyVK_rvi7db")
# %%
MAG = wk.Mag("1")
# %%
SEGMENT_IDS = [1]
# %%
mag_view = dataset2.get_segmentation_layers()[0].get_mag(MAG)

# %%%
z = mag_view.bounding_box.topleft.z
# %%
import numpy as np

# %%
masks = []
with mag_view.get_buffered_slice_reader(buffer_size=2) as reader:
        for slice_data in reader:
            slice_data = slice_data[0]  # First channel only
            for segment_id in SEGMENT_IDS:
                segment_mask = (slice_data == segment_id).astype(
                    np.uint8
                ) * 255  # Make a binary mask 0=empty, 255=segment
                segment_mask = segment_mask.T  # Tiff likes the data transposed
                
                masks.append(segment_mask)
            print(f"Downloaded z={z:04d}")
            z += MAG.z
# %%
view_ = mag_view.get_view()

# %%
the_zarr = mag_view.get_zarr_array()
# %%
import zarr
# %%

remote_dataset = wk.Dataset.open("/home/tmn7/wkw-test/wkw2/binaryData/harvard-htem/dcvsoma323")
# %%
import matplotlib.pyplot as plt
# %%
dataset3 = wkw.Dataset.open('annotations/dcvsoma323_annotation/1')
dataset3.read((0,0,0),(912,912,6))
dataset3.read((0,0,0),(912,912,6))[0,:,:,:]
example = dataset3.read((0,0,0),(912,912,6))[0,:,:,:]
plt.imshow(np.flipud(np.rot90(example[:,:,0])), cmap=plt.cm.gray)

# %% 
from PIL import Image
from tifffile import imread
vol = imread("cutouts/dcvsoma323/img/vol.tiff")
seg = imread("cutouts/dcvsoma323/cellseg/cellseg.tiff")
vol = np.array(vol).T
seg = np.array(seg).T
plt.imshow(np.flipud(np.rot90(vol[:,:,0])), cmap=plt.cm.gray)
# %%
