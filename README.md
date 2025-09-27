# Hydro-Transformer
### Transformer Framework for Streamflow Estimation in Ungaged Basins Using Large-Scale Hydrologic Simulations

This repository provides a **data preprocessing and modeling framework** for training transformer-based models on hydrological datasets. The focus is on **predicting streamflow in ungaged basins** using the **Retrospective National Water Model (NWM)** dataset for the 671 identified **CAMELS basins**.

Our proof-of-concept evaluates the hypothesis that **using both upstream and downstream forcings (combined model)** provides stronger predictive performance than relying on **upstream-only forcings**.

---

## Data Organization

All raw and processed data should be stored inside the `data/` folder. The structure is as follows:
- data/
- ├── time_series/ # Contains forcing data files in .nc or .zarr format
- ├── attributes/ # Contains static attributes CSV for all basins
- └── basins/ # Contains basin text files (IDs) for training/validation/test

- **`time_series/`**  
  Contains NetCDF or Zarr files with time-series forcing data for upstream–downstream basin pairs.  
  Files follow the naming convention:  <downstream>_<upstream>.nc
  Example: `12345678_87654321.nc`

- **`attributes/`**  
    - Contains a CSV with static basin descriptors (e.g., `camelsatts.csv`). These include:  
        - Basin length  
        - Basin area  
        - Reach length  

- **`gages/`**  
Contains text files listing gauge IDs for training, validation, and testing.  

---

## Supporting Datasets

1. **Retrospective National Water Model (NWM)**  
 - Provides multi-decadal hydrologic simulations (precipitation, temperature, humidity, pressure, radiation, streamflow, etc.).  
 - Used to generate dynamic forcing inputs.

2. **CAMELS Dataset**  
 - Provides static attributes and metadata for basins across the U.S.  
 - Required files include:  
   - `camels_link.csv`: maps basin IDs (COMIDs) to USGS gages.  
   - `camelsatts.csv`: contains static attributes for each basin.  

---

## Preprocessing Workflow

Before training models, the dataset must be preprocessed into a format suitable for NeuralHydrology.

### Steps

1. **Basin Pair Preprocessing**  
 Run the script:  01_preprocessing/01_basin_pair_preprocessor.py
 This will:  
 - Extract downstream–upstream basin pairs from filenames.  
 - Map downstream IDs to CAMELS gages via `camels_link.csv`.  
 - Merge and save static attributes from `camelsatts.csv`.  
 - Rename raw `.nc/.zarr` files from `<downstream>_<upstream>.nc` to `<gage>.nc`.

2. **Correlation Analysis**  
Perform exploratory checks on relationships between:  
 - **Dynamic variables** (meteorological forcings, streamflow).  
 - **Static attributes** (basin length, area, reach length).  
Heatmaps are generated to ensure no redundancy or high collinearity dominates inputs.

3. **Model Training with NeuralHydrology**  
After preprocessing, data can be used with the [NeuralHydrology](https://neuralhydrology.github.io/) framework.  
- Configure experiment files (`.yml`) to specify:  
  - Training/validation/test basins.  
  - Input variables (static and dynamic).  
  - Model type (`transformer`).  
  - Hyperparameters (sequence length, layers, heads, feedforward dimensions, etc.).  

---

## Model Concept

- **Inputs:**  
 - Downstream and upstream forcing variables (combined).  
 - Static attributes for each basin.  

- **Outputs:**  
 - Predicted streamflow at the upstream gage.  

- **Hypothesis:**  
 - A combined downstream–upstream input model will **outperform upstream-only models** by leveraging integrated hydrological signals.  

---

## Example Usage

### Step 1. Preprocess Basin Pairs
```bash 
python 01_preprocessing/01_basin_pair_preprocessor.py
```

### Step 2. Run Correlation Analysis
```bash
python 01_preprocessing/correlation_analysis.py
```

### Step 3. Train Model with NeuralHydrology

```bash
python -m neuralhydrology train --config configs/transformer.yml
```

Citation

"..."
