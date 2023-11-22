# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlitedict import SqliteDict
from neurometry import ml
import copy
from torchvision import transforms
import skimage.measure

# %%
sample = pd.read_csv("data/codex_5000sample_slicekey_9_18_23.csv")
df = pd.read_csv("data/codex_5000sample_9_18_23.csv")

db = SqliteDict("data/root_to_mask_9_18_23.sqlite", encode="ascii")

#convert bytes to int
# %%
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

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def get_pad(a):
    _size = a.shape
    l = _size[0]
    w = _size[1]
    l_2 = next_power_of_2(l)
    w_2 = next_power_of_2(w)
    new_size = max(l_2, w_2)
    diff_l = new_size - l
    diff_w = new_size - w

    if diff_l % 2 == 0:
        pad_l = (int(diff_l/2), int(diff_l/2))
    else:
        pad_l = (int(diff_l/2), int(diff_l/2)+1)
    
    if diff_w % 2 == 0:
        pad_w = (int(diff_w/2), int(diff_w/2))
    else:
        pad_w = (int(diff_w/2), int(diff_w/2)+1)
    
    return np.pad(a, [pad_l, pad_w], mode='constant')

#input: padded matrix
def plot_results(matrix, cutoff):
    converted = np.array(matrix/255.0).astype('float32')
    converted_tensor = transforms.ToTensor()(converted)
    result = model(converted_tensor[None, :,:,:])[0,0,:,:].detach().numpy()
    filtered = result > cutoff
    filtered = 1*filtered

    plt.subplots(figsize=(20, 10))
    # using subplot function and creating 
    # plot one
    plt.subplot(1, 2, 1)
    plt.imshow(converted, cmap=plt.cm.gray)
    plt.title('FIRST PLOT')
    # using subplot function and creating plot two
    plt.subplot(1, 2, 2)
    plt.imshow(converted, cmap='gray')
    plt.imshow(filtered, cmap='Reds', alpha=.5)
    #plt.imshow(result, cmap=plt.cm.gray)
    plt.title('SECOND PLOT')
    
    #plt.subplot(1, 3, 3)
    #plt.imshow(filtered, cmap=plt.cm.gray)
    #plt.title('THIRD PLOT')
    # space between the plots
    plt.tight_layout()
    # show plot
    plt.show()


def get_crop(matrix):
    mask = matrix > 0
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    # Get the contents of the bounding box.
    return copy.deepcopy(matrix[x0:x1, y0:y1])

def pipeline(matrix, cutoff):
    cropped = get_crop(matrix)
    padded = get_pad(cropped)
    plot_results(padded, cutoff)

# eccentricity less than .75
# vesicle size greater than 25
def pipeline_for_vesicle_saturation(matrix, cutoff):
    cropped = get_crop(matrix)
    padded = get_pad(cropped)
    converted = np.array(padded/255.0).astype('float32')
    #area
    area =  (converted > 0).sum()
    converted_tensor = transforms.ToTensor()(converted)
    result = model(converted_tensor[None, :,:,:])[0,0,:,:].detach().numpy()
    filtered = result > cutoff
    labeled_image, count = skimage.measure.label(filtered, return_num=True)
    objects = skimage.measure.regionprops(labeled_image)
    final_count = len([obj for obj in objects if obj.area>25 and obj.eccentricity < .75])
    #object_areas = [obj["area"] for obj in objects]
    #object_ecc = [obj["eccentricity"] for obj in objects]
    #return area, count, object_areas, object_ecc
    return area, final_count

def flatten(l):
    return [item for sublist in l for item in sublist]


config = ml.load_config("config.yaml")
final_activation = ml.get_final_activation()
model = ml.UNet(in_channels=1, out_channels=1, depth=config["train"]["depth"], final_activation=final_activation)

ml.load_weights(model, config["train"]["weights_path"]+config["train"]["id"]+".pth")

model.eval()

# %%
database = SqliteDict('data/codex_5000sample_11_15_23.sqlite', autocommit=True)
counter = 0
for ex in df["root_id"]:
    counter += 1
    print(counter)
    if ex not in database:
        try:
            cutout = db[int_to_bytes(ex)]
        except:
            pass
        else:
            z = cutout.shape[2]
            areas = []
            counts = []
            sizes = []
            eccentricities = []
            for i in range(z):
                #empty
                if not np.any(cutout[:,:,i]):
                    areas.append(0)
                    counts.append(0)
                    #sizes.append([])
                    #eccentricities.append([])
                else:
                    #area, count, size, ecc = pipeline_for_vesicle_saturation(cutout[:,:,i], .10)
                    try:
                        area, count = pipeline_for_vesicle_saturation(cutout[:,:,i], .10)
                        areas.append(area)
                        counts.append(count)
                    except:
                        areas.append(0)
                        counts.append(0)
                    #sizes.append(size)
                    #eccentricities.append(ecc)
                
            sum_counts = sum(counts)
            sum_areas = sum(areas)
            database[ex] = [sum_counts, sum_areas]


database.close()

'''
ex = 720575940629248374
cutout = db[int_to_bytes(ex)]
zeroth = cutout[:,:,0]
mask = zeroth > 0
# Coordinates of non-black pixels.
coords = np.argwhere(mask)
# Bounding box of non-black pixels.
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
# Get the contents of the bounding box.
cropped = copy.deepcopy(zeroth[x0:x1, y0:y1])
#plt.imshow(cropped, cmap=plt.cm.gray)
_size = cropped.shape
l = _size[0]
w = _size[1]
l_2 = next_power_of_2(l)
w_2 = next_power_of_2(w)
new_size = max(l_2, w_2)
padded = np.pad(cropped, [(int((new_size-l)/2), int((new_size-l)/2)), (int((new_size-w)/2), int((new_size-w)/2))], mode='constant')
padded = get_pad(cropped)
tester = copy.deepcopy(padded)
# %%
tester = np.array(tester/255.0).astype('float32')
# %%

tester_tensor = transforms.ToTensor()(tester)
plt.imshow(model(tester_tensor[None, :,:,:])[0,0,:,:].detach().numpy(), cmap=plt.cm.gray)
# %%
plt.imshow(model(tester_tensor[None, :,:256,:256])[0,0,:,:].detach().numpy(), cmap=plt.cm.gray)
# %%
result = model(tester_tensor[None, :,:,:])[0,0,:,:].detach().numpy()
result_try = result > .10
labeled_image, count = skimage.measure.label(result_try, return_num=True)
# %%
objects = skimage.measure.regionprops(labeled_image)

# %%
object_areas = [obj["area"] for obj in objects]
object_ecc = [obj["eccentricity"] for obj in objects]
'''