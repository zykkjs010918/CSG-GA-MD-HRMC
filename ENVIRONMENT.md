# Environment Setup and Installation Guide

## 1. Overview

This document describes the software environment, major dependencies, and installation requirements for the coal 3D molecular structure reconstruction program. The program relies on Python scientific computing libraries, a locally modified version of Fullrmc, and LAMMPS with support for the ReaxFF reactive force field.

This guide is based on the environment used during software development and testing. Since operating systems, software versions, and build configurations may vary, adjustments may be required when using a different environment. Compatibility should be verified before performing production calculations.

## 2. Software Environment

### 2.1 Operating System and Runtime Environment

The main software environment used during the development and testing of this program is as follows:

- Operating system: Rocky Linux 10.2
- Python: 3.9.25
- Environment manager: Conda
- Molecular dynamics software: LAMMPS (22 Jul 2025 - Update 3)
- Reactive force field: ReaxFF
- Acceleration support: LAMMPS KOKKOS

### 2.2 Python Dependencies

The program primarily depends on the following Python packages and interfaces:

- NumPy: 1.23.0
- SciPy: 1.13.1
- Matplotlib: 3.9.4
- pdbparser: 0.1.8
- Fullrmc: a locally modified version based on the publicly available source code
- LAMMPS Python interface: compatible with the installed LAMMPS build

The versions listed above represent the intended dependency configuration for the project. Before use, compatibility among the relevant packages and their dependencies should be verified.

## 3. Environment Setup and Dependency Installation

### 3.1 Creating a Python Environment

This program uses Conda to manage its Python runtime environment. A dedicated Python 3.9 environment can be created using the following commands:

```bash
conda create -n coal-reconstruction python=3.9
conda activate coal-reconstruction
```

Here, `coal-reconstruction` is an example environment name and can be changed as needed. If an existing Conda environment is used, activate the corresponding environment instead.

### 3.2 Installing Python Dependencies

After activating the Python environment, install the main scientific computing dependencies:

```bash
pip install numpy==1.23.0 scipy==1.13.1 matplotlib==3.9.4 pdbparser==0.1.8
```

After installation, the following command can be used to check whether the relevant modules can be imported successfully:

```bash
python -c "import numpy, scipy, matplotlib, pdbparser; print('Python dependencies loaded successfully.')"
```

### 3.3 Configuring Fullrmc

This program is developed based on the publicly available Fullrmc source code, with additional modifications and extensions to meet the requirements of coal 3D molecular structure reconstruction.

The original source code and installation instructions are available in the [official Fullrmc GitHub repository](https://github.com/bachiraoun/fullrmc).

Before using this program, install the base package according to the original project's instructions and integrate the additional modules developed for this project into the appropriate source code. Ensure that Python imports the modified version of Fullrmc and verify the compatibility of the relevant modules with the main program.

### 3.4 Configuring LAMMPS and ReaxFF

This program uses LAMMPS to perform energy calculations and molecular dynamics simulations based on the ReaxFF reactive force field. The LAMMPS version used in the development environment is **22 Jul 2025 - Update 3**.

The LAMMPS source code and technical documentation are available in the [official LAMMPS GitHub repository](https://github.com/lammps/lammps).

Before running the program, ensure that the LAMMPS installation provides the following capabilities:

- Support for the ReaxFF reactive force field and the associated charge-equilibration calculations.
- Support for the KOKKOS functionality required by the program.
- A Python interface that can be called successfully from the active Python environment.
- Correct loading of the required shared libraries.

LAMMPS build options, Python interface configuration, and shared-library settings depend on the operating system, compiler, and installation method. Refer to the official LAMMPS documentation for configuration details.

After configuration, verify that the LAMMPS executable, Python interface, and required features function correctly. Test the complete workflow using the actual input files provided for the project to confirm compatibility between LAMMPS and the main program.

## 4. Notes

1. The software versions and dependency configuration listed in this guide are based on the project's development and testing environment. Compatibility should be checked whenever software versions are changed.
2. This program includes local modifications to Fullrmc. Installing the original Fullrmc package alone does not provide all the extended functionality required by this project.
3. The LAMMPS Python interface, shared libraries, and ReaxFF- and KOKKOS-related features must be compatible with the actual runtime configuration.
4. Before performing production calculations, use the project's actual input files to verify structure loading, force-field initialization, energy calculations, and optimization procedures.
5. Computational performance depends on the hardware configuration, model size, and computational parameters. Population size, number of generations, and parallel task settings should be adjusted according to the available computational resources.

