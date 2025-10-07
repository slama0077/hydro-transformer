# Hydro-Transformer
### Transformer Framework for Streamflow Estimation in Ungaged Basins Using Large-Scale Hydrologic Simulations

This repository provides a **data preprocessing and modeling framework** for training transformer-based models on hydrological datasets. The focus is on **predicting streamflow in ungaged basins** using the **Retrospective National Water Model (NWM)** dataset for the 671 identified **CAMELS basins**.

Our proof-of-concept evaluates the hypothesis that **using both upstream and downstream forcings (combined model)** provides stronger predictive performance than relying on **upstream-only forcings**.

---

## Data Organization

All raw and processed data should be stored inside the `data/` folder. The structure is as follows:
- data/n{reach_value} 
- ├── time_series/ # Contains forcing data files in .nc or .zarr format
- ├── attributes/ # Contains static attributes CSV for all basins
- └── gages/ # Contains gages text files (IDs) for training/validation/test
  - ├── `basin_chunk_*.txt` # contains gage chunks for training
  - ├── `gage_list.txt` # conatins full gage list
  - ├── `gage_list_clean.txt` # contains cleaned gage list (after removing invalid gages) 

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
 - Used to generate static attributes.

2. **CAMELS Dataset**  
 - Provides static attributes and metadata for basins across the U.S.  
 - Required files include:  
   - `camels_link.csv`: maps basin IDs (COMIDs) to USGS gages.  
   - `camelsatts.csv`: contains static attributes for each basin.  

---
## Environment Setup 
#### Create conda environment with CUDA 11.8 support

```bash
conda env create -f environment_cuda11_8.yml
```


## Preprocessing Workflow

Before training models, the dataset must be preprocessed into a format suitable for NeuralHydrology.

### Steps

1. Download the time-series files and store in data/n{reach_value} 

```bash
aws s3 cp s3://camels-nwm-reanalysis/n{reach_value}/ data/{reach_value}/time_series --recursive --no-sign-request
```

2. **Basin Pair Preprocessing**  
 Run the script:  
 ```bash 
 python 01_preprocessing/01_basin_pair_preprocessor.py
 ```
 - Extract downstream–upstream basin pairs from filenames.  
 - Rename raw files from `<downstream>_<upstream>.nc` to `<gage>.nc`.

3. **Extract Static Attributes**
Run the script:
```bash
python 01_preprocessing/02_extract_static_attributes.py
```
 - Map downstream IDs to CAMELS gages via `data/camels_link.csv`.  
 - Merge and save static attributes from `data/nwm_attributes.csv`.  

4. **Create Gages Chunks** -- Optional
 Run the script: 
 ```bash 
 python 01_preprocessing/03_create_gages_chunks.py
 ```
 Resulting files are renamed using the prefix 'basin' to simplify file naming conventions.
 - Splits `gage_list.txt` into smaller training chunks.
 - Outputs files like: `gages/basin_chunks_{i}.txt`.
 **Note: Use this option if you plan to train on hourly data, since training across all gages at this resolution will require significantly more time.**

5. **Remove Invalid Gages (after initial; training)**
 Run the script: 
  ```bash 
 python  01_preprocessing/03_remove_invalid_gages.py
 ```
 - Removes invalid gage IDs from chunk files.
 - Recommended after detecting unstable or low-variance gages.

 Can also run:
 ```bash
 python 02_train_configs/01_clean_gage_list.py
 ```
 - Removes invalid gage IDs from whole gage_list file.

6. **Correlation Analysis**  
 Run the script: 
  ```bash 
 python  01_preprocessing/03_correlation_analysis.py
 ```
 Perform exploratory checks on relationships between:  
 - **Dynamic variables** (meteorological forcings, streamflow).  
 - **Static attributes** (basin length, area, reach length).  
 Generates heatmaps to detect redundancy or strong collinearity.

7. **Model Training with NeuralHydrology**  
After preprocessing, data can be used with the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework.  
- Configure experiment files (`02_training/{model}/*.yml`) to specify:  
  - Training/validation/test basins.  
  - Input variables (static and dynamic). 
  - Output (predicted streamflow at upstream gage) 
  - Model type (`transformer` or `lstm`).  
  - Hyperparameters (sequence length, layers, heads, feedforward dimensions, etc.).  

7. **Training the model**
Run the script: `neuralhydrology_py/train.py`
This will start training... look below for example useage.
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
python train.py 02_train_configs/transformer/transformer_combined.yml #training in chunks
```

```bash
python train.py 02_train_configs/transformer/transformer_upstream.yml --mode all --gage-file data/gage_list_clean.txt --epochs 3 #train all gages, specifying the number of epochs
```

OR using slurm run (will run on all gage list)
```bash
sbatch run_train.slurm
```


### Citation
If you wnat to use this repository, please cite as:

`"Hydro-Transformer: Transformer Framework for Streamflow Estimation in Ungauged Basins Using Large-Scale Hydrologic Simulations"`

