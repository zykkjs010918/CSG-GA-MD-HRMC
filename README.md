# Molecular Structure Reconstruction of Bituminous Coal Based on a Carbon-Skeleton-Guided Genetic Algorithm and MD-HRMC

## 1. Project Overview

This project provides the core code for a computational method for three-dimensional molecular structure reconstruction of bituminous coal. The method combines a Carbon-Skeleton-Guided Genetic Algorithm (CSG-GA) with Molecular Dynamics-Hybrid Reverse Monte Carlo (MD-HRMC) to achieve global search and local optimization of candidate structures.

CSG-GA uses the carbon skeleton as the structural basis for genetic operations and generates candidate configurations through a crossover operator (CSG-C) and a mutation operator (CSG-M), while accounting for local structural relationships within coal molecules during genetic search. MD-HRMC combines molecular dynamics relaxation using the ReaxFF reactive force field with hybrid reverse Monte Carlo sampling to locally optimize candidate structures.

By periodically introducing MD-HRMC during genetic algorithm evolution and feeding the optimized structures back into the genetic search, the method coordinates global configurational exploration with local structural refinement.

## 2. Methodology

### 2.1 Carbon-Skeleton-Guided Genetic Algorithm (CSG-GA)

CSG-GA uses the carbon skeleton as the structural basis for genetic operations and generates candidate configurations using the crossover operator (CSG-C) and mutation operator (CSG-M). Both operators employ a hierarchical strategy based on element categories and incorporate atomic spatial positions and local structural environments into structural adjustments.

- **CSG-C:** Recombines parental carbon skeletons using OT. Rigid-body alignment, independent OT displacements, and Gaussian-kernel interpolation are further used to coordinate the spatial adjustment of heteroatoms and hydrogen atoms, enabling the exchange of structural information between different parent structures.
- **CSG-M:** Applies spatially smooth random deformation to a single parental carbon skeleton and uses OT to determine the displacement of carbon atoms during mutation. Heteroatom and hydrogen configurations are adjusted through rigid-body alignment, controlled random perturbations, and Gaussian-kernel interpolation to increase population diversity.

Both operators use a staged parameter-annealing strategy. CSG-GA performs global search through population evolution and evaluates candidate structures based on PDF fitting error and system energy.

### 2.2 MD-HRMC Local Optimization

MD-HRMC combines ReaxFF molecular dynamics relaxation with hybrid reverse Monte Carlo structural sampling to locally optimize candidate configurations under constraints imposed by system energy and experimental structural information. The optimized structures are fed back into the genetic algorithm population for subsequent global search, enabling local structural refinement and renewed global exploration.

## 3. Project Files

The project mainly includes the following core code, configuration documentation, and input files:

`GA_parallel.py` - Main genetic algorithm program for candidate structure generation, evaluation, and population evolution.
`MDHRMCEngine.py` - MD-HRMC local optimization engine.
`ENVIRONMENT.md` - Software operating environment and dependency description.
`coal_model.gr` - Target PDF data.
`coal_modelchu.pdb` - Initial coal molecular structure model.
`coal-HCONSB.ff` - ReaxFF force-field parameter file.
`lmp_control` - LAMMPS control parameter file.
`.gitignore` - Git ignore rules.

**Note:** The actual list of publicly released files will be determined according to code provenance, file licensing, and data-management requirements. For third-party code or input files that cannot be released at this stage, information on how to obtain them will be provided where appropriate.

## 4. Software Environment and Dependencies

The program was developed and run in a Linux environment. The main verified computational environment is Rocky Linux 10.2 with Python 3.9.25. The program depends on NumPy, SciPy, Matplotlib, pdbparser, a locally modified version of fullrmc, and the LAMMPS Python interface.

The local version of fullrmc has been modified and extended based on its publicly available code. The original project is available at [bachiraoun/fullrmc](https://github.com/bachiraoun/fullrmc).

LAMMPS must support ReaxFF and be configured with a compatible Python interface. The program also uses KOKKOS-related runtime options; therefore, the runtime environment must meet the corresponding compilation and hardware requirements.

For detailed software versions and dependency configurations, see [ENVIRONMENT.md](ENVIRONMENT.md).

## 5. Running the Program

After configuring the runtime environment and preparing the required input files, run the following command from the project working directory to start the program:

```bash
nohup python GA_parallel.py > GA.log 2>&1 &
```

The program reads input files from the current working directory and generates the corresponding computational results during execution.

## 6. Reproducibility

This project makes the core algorithm implementation for three-dimensional molecular structure reconstruction of bituminous coal available to support inspection and research on the method. Full computational reproduction also requires the corresponding input data, algorithm parameters, and software environment configuration. The availability of related materials will be determined in accordance with data-management requirements and permissions for public release.
