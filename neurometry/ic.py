import imageryclient as ic
import tifffile
import matplotlib.pyplot as plt
import numpy as np

def source():
    return ic.ImageryClient(image_source = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14' ,
                 segmentation_source='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
                 image_mip=1, base_resolution=[4,4,40])

def get_img_and_seg(cutout, bounds, pre_syn):
    imgvol, segdict = cutout.image_and_segmentation_cutout(bounds,
                                                   split_segmentations=True, root_ids=[pre_syn])
    return imgvol, segdict

def get_overlay(imgvol, segdict):

    return ic.composite_overlay(segdict, imagery=imgvol)

def plotter(pre_syn, post_syn, imgvol, segdict):
    f , ax = plt.subplots(16,3, figsize=(10,20))
    # lets loop over z sections
    for i in range(16):
        # plot the images in column 0
        ax[i, 0].imshow(np.squeeze(imgvol[:,:,i]),
                        cmap=plt.cm.gray,
                        vmax=255,
                        vmin=0)
        # plot the pre-synaptic mask in column 1
        ax[i, 1].imshow(np.squeeze(segdict[pre_syn][:,:,i]))
        # plot the post-synaptic mask in column 2
        ax[i, 2].imshow(np.squeeze(segdict[post_syn][:,:,i]))
    f.tight_layout()

def writer(name, img):
    tifffile.imwrite(name, data=img)

def stacker(overlays):
    #quick view of the stacked overlays
    return ic.stack_images(overlays)

def create_bounds(coordinates, xysize, zsize):
    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]

    xysize = xysize // 2
    zsize = zsize // 2
    bounds=[[x-xysize, y-xysize, z-zsize],
        [x+xysize, y+xysize, z+zsize]]

    return bounds