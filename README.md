# vesicleseg
DCV segmentation pipeline including data ingestion, preprocessing, and U-Net model deployment and inference of FAFB dataset

## Run

```bash
python pipeline.py
```

## Main Operations

**download.py** 
  
  - Downloads DCV annotations from WebKnossos, the digital platform we use for painting cellular features 

**pipeline.py**

  - Driver code that reads in customizable model configuration yaml file to deploy U-Net, saving model weights

**inferencer.py**
  
  - Loads model weights to predict DCV masks from new data

**function.py**

  - Input: A Root ID
  - Output: Vesicle Saturation Metric
  - Procedure: Use existing nuclei segmentation of root id to estimate bounding box encompassing soma, create sections, filter, apply working model to each section, count up segmented bodies, divide by total area seen

**cca.py**

  - Connected component analysis, allows for calculation of size and eccentricity of segmented vesicles in 2D and 3D inputs

## Model Configuration File
  Customizable in that you can choose hyperparameters, edit here: **input/config.yaml**

```jsx
dataset:
  ann_path: unknown
  vol_path: unknown
optimizer:
  choice: Adam
  initial_lr: 0.0001
train:
  logger: True
  id: UnetA_bs64t1_model
  logger_path: runs/
  weights_path: outputs/weights/
  batch_size: 16
  epochs: 1000
  depth: 4
  metric: Dice Coefficient
  test_size: .2
  random_state: 49
```

## Results

Peptidergic neuron from pars intercerabrilis segmentation

![endocrine1](https://github.com/azatian/vesicleseg/assets/9220290/3767b4af-be98-4012-9f96-6ea3f62b9df2)
![endocrine2](https://github.com/azatian/vesicleseg/assets/9220290/f9920d8c-1194-4505-a6e5-37f47202acfe)



