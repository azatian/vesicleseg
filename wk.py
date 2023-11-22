# %%
import webknossos as wk
import sys

# %%
from webknossos.dataset.properties import (
    DatasetViewConfiguration,
    LayerViewConfiguration,
)
# %%
#dataset = wk.Dataset.from_images(input_path="cutouts/dcvsyn1_source",
#                                output_path="cutouts/dcvsyn1wk",
#                                voxel_size=(4,4,40),
#                                name="img")
def main() -> None:

    args = sys.argv[1:]
    syn = args[0]
    
    wk_filepath = "/home/tmn7/wkw-test/wkw2/binaryData/harvard-htem/"

    dataset = wk.Dataset.from_images(input_path="cutouts/" + syn + "/img",
                                output_path=wk_filepath + syn,
                                voxel_size=(4,4,40),
                                name=syn)
    #dataset = wk.Dataset("cutouts/dcvsyn1",
    #                    voxel_size=(4,4,40))

    dataset.layers['vol.tiff'].name = "img"

    dataset_pre = wk.Dataset.from_images(input_path="cutouts/" + syn + "/cellseg",
                                output_path=wk_filepath + syn,
                                voxel_size=(4,4,40),
                                name=syn)
    

    dataset_pre.layers['cellseg.tiff'].default_view_configuration = LayerViewConfiguration(color=(184, 66, 66), alpha=15)
    dataset_pre.layers['cellseg.tiff'].name = "cellseg"


    # %%
    #img_layer = dataset.add_layer_from_images(images="cutouts/dcvsyn1_source",
     #                               layer_name="img", dtype='uint8', channel=0)

    # %%
    #pre_layer = dataset.add_layer_from_images(images="cutouts/dcvsyn1_presyn",
     #                                          layer_name="presyn", dtype='uint8', channel=1)

    #post_layer = dataset.add_layer_from_images(images="cutouts/dcvsyn1_postsyn",
     #                                           layer_name="postsyn", dtype='uint8', channel=0)


    # %%
    #pre_layer.default_view_configuration = LayerViewConfiguration(
    #        color=(184, 66, 66), alpha=20)

    #post_layer.default_view_configuration = LayerViewConfiguration(
     #       color=(57, 81, 137), alpha=20)

# %%
if __name__ == "__main__":
    main()