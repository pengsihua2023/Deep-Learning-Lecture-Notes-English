## Libraries Related to PINNs

Physics-Informed Neural Networks (PINNs) libraries and frameworks are mainly used in deep learning and scientific computing for solving partial differential equations (PDEs), ordinary differential equations (ODEs), and other physics-based modeling problems. Commonly used libraries include:

### Mainstream PINN Libraries

* **DeepXDE**
  A TensorFlow- and PyTorch-based PINN library that supports ODEs, PDEs, integral equations, and more. Comprehensive features, widely adopted in academia.
  [DeepXDE GitHub](https://github.com/lululxvi/deepxde)

* **NVIDIA Modulus (formerly SimNet)**
  NVIDIA’s industrial-grade PINN framework, supporting applications in CFD, structural mechanics, electromagnetics, etc., optimized for GPUs.
  [Modulus GitHub](https://github.com/NVIDIA/modulus)

* **SciANN**
  A Keras-based PINN library suitable for rapid prototyping, mainly used for PDE/ODE solving.
  [SciANN GitHub](https://github.com/ehsanhaghighat/sciann)

* **NeuralPDE (Julia)**
  A Julia-based PINN framework, part of the SciML ecosystem, well-suited for research and numerical computation.
  [NeuralPDE GitHub](https://github.com/SciML/NeuralPDE.jl)

### Support within Broader Deep Learning Ecosystems

* **PyTorch Lightning + PINN Templates**
  Community-provided templates for training PINNs within the Lightning framework.

* **JAX-based PINN Implementations**
  Cutting-edge research implementations leveraging JAX for automatic differentiation and GPU/TPU parallelization.

---

The following comparison table summarizes commonly used PINN libraries across **language/backend, domain strengths, equation/constraint support, geometry/data handling, training/engineering support, ecosystem maturity**, and more — to help with selection.

| Library / Framework                         | Language / Backend           | Main Focus & Strengths                                        | Equation / Constraint Support                              | Geometry / Mesh & Data Interface                         | Training & Engineering                                                 | Ecosystem / Maturity                    | License        | Suitable Users / Scenarios                                     |
| ------------------------------------------- | ---------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------- | -------------- | -------------------------------------------------------------- |
| **DeepXDE**                                 | Python; TF/PyTorch           | General PINNs: ODE, PDE, integral eqs, inverse                | BC/IC, measurement points, variational forms               | Built-in geometries; import point cloud/boundary samples | Single-machine; callbacks, adaptive sampling, loss weighting           | Widely used in academia, tutorials      | MIT            | Academic prototyping, paper reproduction                       |
| **NVIDIA Modulus (formerly SimNet)**        | Python; PyTorch + CUDA       | Industrial-scale PINN / multiphysics (CFD, heat transfer, EM) | Strong/weak forms, constraint composition, parameter scans | Rich geometry constructors; mesh/point cloud/CFD data    | Multi-GPU, mixed precision, deployment ready                           | Many industrial cases, actively updated | BSD-3          | Industrial deployment, GPU performance & scaling               |
| **SciANN**                                  | Python; Keras/TensorFlow     | Lightweight PINN prototyping                                  | PDE/ODE, data-driven constraints                           | Point samples; simple geometry                           | Easy to use, Keras-like API                                            | Relatively lightweight                  | MIT            | Teaching, course experiments, quick trials                     |
| **NeuralPDE.jl (SciML)**                    | Julia; AD & DiffEq ecosystem | Research & numerical methods (PINN, DeepONet, hybrids)        | Flexible weak/variational forms, constraint composition    | Integrated with DifferentialEquations.jl/FEM tools       | Supports parallelism, parameter estimation, uncertainty quantification | Deep academic integration               | MIT            | Researchers needing composability & advanced numerical methods |
| **JAX Implementations (e.g. Equinox/Flax)** | Python; JAX / Accelerator    | Cutting-edge research prototypes, efficient TPU/GPU AD        | Highly customizable losses/constraints                     | Custom-built geometries & sampling required              | pmap/vmap parallelism, compilation optimization                        | Fragmented but active                   | Apache/MIT     | Researchers needing “high customizability + high performance”  |
| **PyTorch Lightning PINN Templates**        | Python; PyTorch              | Engineering-friendly training workflows                       | Depends on template implementation                         | User-organized                                           | Logging, callbacks, distributed training                               | Many community templates                | —              | Users integrating PINNs into existing ML stacks                |
| **FEniCS/FEniCSx + PINN Hybrids**           | Python/C++; FEM ecosystem    | PINNs + FEM coupling, weak-form PDEs                          | Strong/weak forms, detailed PDE constraints                | Mature mesh/solvers; good for complex geometries         | Integrated with FEM workflows                                          | Useful for research & engineering       | LGPL           | High-accuracy PDEs, complex geometries & boundaries            |
| **TFPINN / Community TF Implementations**   | Python; TensorFlow           | Educational demos                                             | Basic PDE/ODE support                                      | Point-based sampling                                     | Lightweight, teaching-friendly                                         | Scattered, inconsistent updates         | Varies by repo | Beginners, classroom demonstrations                            |

---

