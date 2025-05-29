# Quantum-Enhanced Multi-Objective Optimization for Drug Discovery

This repository contains the experimental code and resources for the research paper: "Quantum-Enhanced Multi-Objective Optimization for Drug Discovery: Binding Affinity Prediction and Formulation Design."

**Author:** Mostafijur Rahman Mamun
**Date:** May 2025 (or current date of research)

## Abstract

We present a hybrid quantum-classical framework for drug discovery, combining variational quantum machine learning (QML) for binding affinity prediction and quantum annealing for Pareto-optimal formulation design. Our QML model demonstrates improved accuracy over classical baselines for predicting binding affinities using a 4-qubit variational circuit. Concurrently, our Quadratic Unconstrained Binary Optimization (QUBO) model, solved via D-Wave's quantum annealer, efficiently identifies Pareto-optimal drug formulations by balancing bioavailability, toxicity, and stability. This integrated approach highlights the potential of NISQ-era quantum algorithms to accelerate and refine critical stages of the drug discovery pipeline.

## Overview

This research explores a novel two-phase quantum-enhanced workflow for addressing critical challenges in drug discovery:

1.  **Binding Affinity Prediction:** A 4-qubit Variational Quantum Circuit (VQC) implemented with PennyLane is used to predict molecular binding affinities. This QML model is benchmarked against classical machine learning models (Random Forest and Support Vector Regression).
2.  **Drug Formulation Optimization:** A Quadratic Unconstrained Binary Optimization (QUBO) model is formulated to optimize drug formulations by considering multiple objectives: maximizing bioavailability while minimizing toxicity and instability. The QUBO is designed for 20 excipient variables and solved using D-Wave's quantum annealing platform. The performance is compared against classical simulated annealing.

           # Conda environment file (or requirements.txt)
## Getting Started

### Prerequisites

* Python 3.8+
* Access to Google Colab or a local Jupyter environment.
* (Optional) D-Wave Leap account for running QUBO experiments on actual D-Wave hardware.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/quantum-drug-discovery.git](https://github.com/yourusername/quantum-drug-discovery.git)
    cd quantum-drug-discovery
    ```

2.  **Set up the Python environment:**
    It's recommended to use a virtual environment.

    * **Using Conda:**
        ```bash
        conda env create -f environment.yml
        conda activate quantum-drug-discovery-env
        ```
    * **Using pip with `requirements.txt` (Create this file based on imports in notebooks):**
        ```bash
        pip install -r requirements.txt
        ```
    Key libraries include:
    * `pennylane`
    * `numpy`
    * `scikit-learn`
    * `rdkit-pypi` (for molecular descriptors)
    * `dwave-ocean-sdk`
    * `matplotlib`
    * `pandas`

### Data Preparation

* **Davis Dataset (for QML):** Instructions for obtaining and preprocessing the Davis dataset should be followed as per `data/README.md` or within the `QML_Binding_Affinity.ipynb` notebook. This typically involves downloading the dataset, extracting relevant protein-ligand pairs, calculating molecular descriptors (e.g., using RDKit), and performing dimensionality reduction (e.g., PCA) if necessary.
* **Excipient Data (for QUBO):** The QUBO model requires numerical coefficients for bioavailability ($b_i$), pairwise toxicity ($t_{ij}$), and stability ($s_i$) for each of the 20 excipients. The methodology for deriving these coefficients should be detailed in the research paper and implemented or loaded as per `data/README.md` or the `QUBO_Formulation_Optimization.ipynb` notebook.

### Running the Experiments

The primary experiments are structured within Jupyter notebooks located in the `notebooks/` directory:

1.  **QML for Binding Affinity Prediction:**
    * Open and run `notebooks/QML_Binding_Affinity.ipynb`.
    * This notebook covers:
        * Data loading and preprocessing.
        * Implementation of the 4-qubit Variational Quantum Circuit (VQC) using PennyLane.
        * Training the VQC model.
        * Implementation, training, and evaluation of classical Random Forest and SVR models.
        * Comparison of MSE and R² metrics.

2.  **QUBO for Drug Formulation Optimization:**
    * Open and run `notebooks/QUBO_Formulation_Optimization.ipynb`.
    * This notebook covers:
        * Defining the 20-variable QUBO problem based on bioavailability, toxicity, and stability coefficients.
        * Solving the QUBO using D-Wave samplers (e.g., `SimulatedAnnealingSampler` for local simulation or `DWaveSampler`/`LeapHybridSampler` for cloud access).
        * Implementation and execution of classical Simulated Annealing for comparison.
        * Analysis of results, including identification of Pareto-optimal solutions.
    * **Note:** To run on actual D-Wave QPUs, you will need to configure your D-Wave API token as instructed in the D-Wave Ocean SDK documentation.

## Expected Results

* **QML Model:** The VQC model is expected to achieve an MSE of approximately $0.015 \pm 0.002$ and an R² of $0.87$ on the Davis dataset, outperforming classical RF and SVR models (as per Table 1 in the research paper).
* **QUBO Optimization:** The D-Wave implementation is expected to solve 20-variable QUBOs approximately five times faster than a classical simulated annealing approach (this claim requires rigorous benchmarking as detailed in the paper). The process should identify a set of Pareto-optimal formulations.

## Citation

If you use this work or code, please cite the original research paper:
