# DeepMD-kit for Simulating Macroscopic Thermodynamic Processes of Copper Metal
A typical real-data example of using DeepMD-kit in materials science is the simulation of copper (Cu) metal. This example is based on a study of large-scale molecular dynamics (MD) simulations, where a Deep Potential (DP) model is trained using ab initio (first-principles) data and then applied on supercomputers for high-accuracy MD simulations. This approach is suitable for studying thermodynamic properties, defect behavior, and other characteristics of metallic materials.

## 📖 1. Data Source
- The data originates from an ab initio training dataset generated using a concurrent learning scheme, ensuring uniform accuracy across a wide range of thermodynamic conditions.
- The dataset includes local environment descriptors, energies, and forces for copper atoms, based on quantum mechanical calculations (e.g., DFT), covering relevant configuration spaces.

## 📖 2. Model Training Process
- Training is implemented using the DeePMD-kit package, typically taking several hours to a week on a single GPU, depending on data complexity.
- The model employs a deep neural network (DNN) to fit high-dimensional functions, incorporating symmetry constraints and concurrent learning to minimize dataset requirements.
- Example command (assuming the data directory is `cu_data`):
  ```
  dp train input.json
  ```
- Example input configuration file `input.json` (simplified, tailored for the copper system, using the `se_a` descriptor):
  ```json
  {
      "model": {
          "type_map": ["Cu"],
          "descriptor": {
              "type": "se_a",
              "rcut": 8.0, // Cutoff radius 8 Å
              "rcut_smth": 0.5,
              "sel": [512], // Maximum number of neighbor atoms 512
              "neuron": [32, 64, 128] // Embedding network size
          },
          "fitting_net": {
              "neuron": [240, 240, 240], // Fitting network size
              "resnet_dt": true
          }
      },
      "learning_rate": {
          "type": "exp",
          "start_lr": 0.001,
          "stop_lr": 1e-8
      },
      "loss": {
          "start_pref_e": 0.1,
          "limit_pref_e": 1,
          "start_pref_f": 1000,
          "limit_pref_f": 1
      },
      "training": {
          "systems": ["cu_data"],
          "batch_size": "auto",
          "numb_steps": 1000000
      }
  }
  ```
- Freeze the model after training:
  ```
  dp freeze -o frozen.pb
  ```

## 📖 3. MD Simulation Details
- **System Scale**: Scalable up to 127.4 million atoms (127,401,984 atoms), with weak scaling tests from 7.96 million to 127.4 million atoms.
- **Simulation Settings**:
  - Time step: 1.0 fs.
  - Integration scheme: Velocity-Verlet.
  - Initial temperature: 330 K (Boltzmann distribution).
  - Neighbor list update: Every 50 steps, with a 2 Å buffer.
  - Number of steps: 500 steps in the example, but extensible to nanosecond-scale simulations.
- **Performance** (on the Summit supercomputer, 4560 nodes):
  - Double precision: 91 PFLOPS, time per step per atom: 8.1 × 10⁻¹⁰ s.
  - Mixed single precision: 162 PFLOPS, time per step per atom: 4.6 × 10⁻¹⁰ s.
  - Mixed half precision: 275 PFLOPS, time per step per atom: 2.7 × 10⁻¹⁰ s.
  - For a 127.4 million atom system, a nanosecond simulation can be completed in 29 hours (double precision).
- **LAMMPS Integration Example Script** (`in.lammps`, assuming the frozen model is `frozen.pb`):
  ```
  units metal
  boundary p p p
  atom_style atomic
  read_data cu_conf.lmp # Initial copper crystal configuration file
  pair_style deepmd frozen.pb
  pair_coeff * *
  mass 1 63.546 # Cu atomic mass
  timestep 0.001
  thermo 50
  fix 1 all nvt temp 330.0 330.0 0.1
  run 500 # Example run for 500 steps
  ```
- **Run Command** (requires LAMMPS with DeepMD plugin installed):
  ```
  mpirun -np 4 lmp -i in.lammps
  ```

This example demonstrates the efficiency of DeepMD-kit in handling metallic systems in materials science and can be extended to other materials such as alloys or semiconductors. For more details, refer to the `examples` folder in the DeepMD-kit GitHub repository.
