
# Molecular Dynamics Simulation of Gas-Phase Water Molecules (H₂O) Using DeepMD-kit
DeepMD-kit is commonly used in quantum chemistry to simulate systems like gas-phase water molecules (H₂O). This example leverages ab initio molecular dynamics (AIMD) data, generated using density functional theory (DFT) calculations (e.g., via VASP or ABACUS software), to train a Deep Potential (DP) model for efficient simulations with quantum chemical accuracy. The data includes atomic coordinates, forces, energies, and other properties, covering vibrational and rotational modes of water molecules, often used to study hydrogen bond dynamics or reaction pathways.

## 📖 1. Data Source
- The data is derived from ab initio trajectories, such as AIMD simulations of water molecules generated from VASP’s OUTCAR file (approximately 200 frames at 300 K). This ensures quantum chemical accuracy (DFT level, e.g., PBE functional).
- Command to download example data (if available or prepared from tutorials):
  ```
  wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/H2O.tar
  tar xvf H2O.tar
  ```
- Data includes: atomic types (O and H), coordinates, forces, energies, and virial tensors.

## 📖 2. Data Preparation
Using the `dpdata` package, ab initio data is converted to DeepMD-kit format and split into a training set (approximately 160 frames) and a validation set (40 frames).

### Python Code (Data Preparation):
```python
import dpdata
import numpy as np

# Load ab initio data from VASP OUTCAR file
data = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')
print('# the data contains %d frames' % len(data))

# Randomly select 40 frames for validation data
index_validation = np.random.choice(200, size=40, replace=False)

# Use the remaining frames for training data
index_training = list(set(range(200)) - set(index_validation))
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

# Export training data to directory "training_data"
data_training.to_deepmd_npy('training_data')

# Export validation data to directory "validation_data"
data_validation.to_deepmd_npy('validation_data')

print('# the training data contains %d frames' % len(data_training))
print('# the validation data contains %d frames' % len(data_validation))
```

**Explanation**: Output files include `box.npy`, `coord.npy`, `energy.npy`, `force.npy`, etc. The atomic type file `type.raw` example: `0 1 1` (O as 0, two H as 1).

## 📖 3. Model Configuration (JSON Input File: input.json)
Create an `input.json` file tailored for the water molecule system, using the DeepPot-SE descriptor (`se_e2_a` type):
```json
{
    "model": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.00,
            "rcut_smth": 0.50,
            "sel": [2, 1],
            "neuron": [25, 50, 100],
            "resnet_dt": false,
            "axis_neuron": 16,
            "seed": 1,
            "_comment": "that's all"
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
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
            "systems": ["../training_data"],
            "batch_size": "auto",
            "_comment": "that's all"
        },
        "validation_data": {
            "systems": ["../validation_data"],
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

**Explanation**: `type_map` specifies atomic types (O and H); training runs for 100,000 steps, with the loss function focusing on energy and forces.

## 📖 4. Model Training
Start training with the following command:
```
dp train input.json
```
To restart training:
```
dp train --restart model.ckpt input.json
```

**Explanation**: Generates an `lcurve.out` file recording the loss curve (e.g., energy error ~1 meV, force error ~100 meV/Å).

## 📖 5. Freezing and Compressing the Model
Freeze the model:
```
dp freeze -o graph.pb
```
Compress the model (optional, for improved efficiency):
```
dp compress -i graph.pb -o graph-compress.pb
```

## 📖 6. Testing the Model
Test the compressed model on validation data:
```
dp test -m graph-compress.pb -s ../validation_data -n 40 -d results
```

**Explanation**: Outputs RMSE/MAE metrics to verify quantum chemical accuracy.

## 📖 7. Molecular Dynamics Simulation (Integrated with LAMMPS)
Use the trained model for AIMD surrogate simulations. Example LAMMPS input script `in.lammps` (NVT ensemble, 300 K):
```
units metal
boundary p p p
atom_style atomic
read_data conf.lmp # Initial water molecule configuration
pair_style deepmd graph.pb
pair_coeff * *
mass 1 15.999 # O atomic mass
mass 2 1.00794 # H atomic mass
timestep 0.001
thermo 100
fix 1 all nvt temp 300.0 300.0 0.1
run 10000
dump 1 all custom 100 h2o.dump id type x y z
```

Example initial `conf.lmp`:
```
3 atoms
2 atom types
-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi
Masses
1 15.999 # O
2 1.00794 # H
Atoms
1 1 0.000000 0.000000 0.000000
2 2 0.757000 0.586000 0.000000
3 2 -0.757000 0.586000 0.000000
```

Run command:
```
lmp -i in.lammps
```

**Explanation**: Generates a trajectory file `h2o.dump`, which can be used to analyze water molecule vibrational modes, consistent with quantum chemical calculations.

This example can be extended to larger quantum chemical systems, such as solvation effects or reaction simulations. For more details, refer to the `examples` folder in the DeepMD-kit GitHub repository.
