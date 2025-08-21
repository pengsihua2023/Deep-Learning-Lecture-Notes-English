## DeepMD-kit Library for Molecular Dynamics Simulation
The following is an example code implementation using DeepMD-kit, based on real data (using a gaseous methane molecule as an example). This example is sourced from the official DeepMD-kit tutorial and uses ab-initio molecular dynamics data (OUTCAR files) generated from VASP as the real data source. The process includes data preparation, model training, freezing, testing, and other steps.

### 1. Data Preparation
First, you need to download and extract the methane data (real ab-initio trajectory data). Then, use Python code to convert the data into DeepMD-kit format and split it into training and validation sets.

#### Download Data Command:
```
wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/CH4.tar
tar xvf CH4.tar
```

#### Python Code (Data Preparation, Using dpdata Package):
```python
import dpdata
import numpy as np
# Load data from VASP OUTCAR file (real ab-initio MD trajectory)
data = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')
print('# the data contains %d frames' % len(data))
# Randomly select 40 frames as validation data
index_validation = np.random.choice(200, size=40, replace=False)
# Use the remaining frames as training data
index_training = list(set(range(200)) - set(index_validation))
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)
# Export training data to the "training_data" directory
data_training.to_deepmd_npy('training_data')
# Export validation data to the "validation_data" directory
data_validation.to_deepmd_npy('validation_data')
print('# the training data contains %d frames' % len(data_training))
print('# the validation data contains %d frames' % len(data_validation))
```
**Notes**: The data includes atomic types, coordinates, forces, energy, etc., totaling 200 frames. The training set contains approximately 160 frames, and the validation set contains 40 frames. After export, directories will contain files such as `box.npy`, `coord.npy`, `energy.npy`, `force.npy`, etc.

### 2. Model Configuration (JSON Input File: input.json)
Create a file named `input.json` with the following content (using the DeepPot-SE descriptor, suitable for the methane system):
```json
{
    "model": {
        "type_map": ["H", "C"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.00,
            "rcut_smth": 0.50,
            "sel": [4, 1],
            "neuron": [10, 20, 40],
            "resnet_dt": false,
            "axis_neuron": 4,
            "seed": 1,
            "_comment": "that's all"
        },
        "fitting_net": {
            "neuron": [100, 100, 100],
            "resnet_dt": true,
            "seed": 1,
            "_comment": "that's all"
        },
        "_comment": "that's all"
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-8,
        "_comment": "that's all"
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment": "that's all"
    },
    "training": {
        "training_data": {
            "systems": ["../00.data/training_data"],
            "batch_size": "auto",
            "_comment": "that's all"
        },
        "validation_data": {
            "systems": ["../00.data/validation_data/"],
            "batch_size": "auto",
            "numb_btch": 1,
            "_comment": "that's all"
        },
        "numb_steps": 100000,
        "seed": 10,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "save_freq": 10000
    }
}
```
**Notes**: `type_map` specifies atomic types (H and C); the descriptor uses the `se_e2_a` type; the training step count is 100,000.

### 3. Train the Model
Start training with the following command:
```
dp train input.json
```
To restart from a checkpoint:
```
dp train --restart model.ckpt input.json
```
**Notes**: Training will generate a `lcurve.out` file to record the loss curve.

### 4. Freeze and Compress the Model
Freeze the model:
```
dp freeze -o graph.pb
```
Compress the model (optional, for improved efficiency):
```
dp compress -i graph.pb -o graph-compress.pb
```

### 5. Test the Model
Test the compressed model on validation data:
```
dp test -m graph-compress.pb -s ../00.data/validation_data -n 40 -d results
```
**Notes**: `-n 40` specifies testing 40 frames, with output including RMSE/MAE metrics for energy and forces.

### 6. Visualize Learning Curve (Optional Python Code)
```python
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("lcurve.out", names=True)
for name in data.dtype.names[1:-1]:
    plt.plot(data['step'], data[name], label=name)
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
```
**Notes**: Plots the loss curve during training.

This example is based on real ab-initio data and can be run directly (requires installation of DeepMD-kit and dpdata). For examples with water molecules or other systems, refer to the `/examples` folder on the DeepMD-kit GitHub.
