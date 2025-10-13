# mee-ept-sdm

A repository for modeling, data preparation, and variable engineering related to species distribution modeling (SDM) with a focus on aquatic insects (EPT: Ephemeroptera, Plecoptera, Trichoptera) in the context of the MEE project.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Main Features](#main-features)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

This repository provides a pipeline for:
- Creating variables and extracting features from geospatial and tabular data
- Generating patches and remote sensing data for modeling
- Training and evaluating machine learning models (Random Forest, Custom CNN) for SDM tasks

## Repository Structure

```
├── modelling/
│   ├── Custom_CNN_plecoptera_example.ipynb
│   ├── RF_plecoptera_example.ipynb
│   ├── config.py
│   ├── emissions.csv
│   ├── image_preprocessing.py
│   ├── inference.py
│   ├── labels.csv
│   ├── models.py
│   ├── utils.py
├── patch_creation/
│   ├── patch_creation.py
│   ├── remote_sensing_utils.py
├── variable_creation/
│   ├── derived_raster_extraction.py
│   ├── derived_tabular_variables_extraction.py
│   ├── utils.py
├── requirements.txt
├── environment.yml
```

### Key Directories

- **modelling/**: Notebooks and scripts for model training, inference, configuration, and preprocessing.
- **patch_creation/**: Scripts for creating spatial data patches and working with remote sensing data.
- **variable_creation/**: Scripts for extracting and engineering features from raster and tabular sources.

## Main Features

- **End-to-end SDM pipeline**: From raw data to predictions and evaluation.
- **Custom and standard models**: Includes Random Forest and CNN approaches.
- **Geospatial processing**: Patch creation and raster variable extraction utilities.
- **Reproducible environments**: Provided via `environment.yml` and `requirements.txt`.

## Getting Started

Clone the repository:
```bash
git clone https://github.com/augustin-delabrosse/mee-ept-sdm.git
cd mee-ept-sdm
```

Set up the environment (recommended):
```bash
conda env create -f environment.yml
conda activate test-env
```
Or install requirements using pip:
```bash
pip install -r requirements.txt
```

## Usage

- Explore notebooks under `modelling/` for training and evaluating models.
- Use scripts in `patch_creation/` and `variable_creation/` for data preparation and feature extraction.

## Requirements

See [requirements.txt](https://github.com/augustin-delabrosse/mee-ept-sdm/blob/main/requirements.txt) or [environment.yml](https://github.com/augustin-delabrosse/mee-ept-sdm/blob/main/environment.yml) for the full list of dependencies.

## Acknowledgements

This project is part of the MEE initiative and leverages open-source geospatial and machine learning libraries.

---
For more details, see the individual script and notebook documentation.