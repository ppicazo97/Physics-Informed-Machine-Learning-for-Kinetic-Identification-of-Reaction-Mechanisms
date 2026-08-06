# ML-RKIM: Physics-Informed Machine Learning for Reaction Kinetics Identification and Modeling

This repository contains the code, datasets, and supporting files used for the ML-RKIM framework presented in the paper. ML-RKIM is a physics-informed machine learning framework developed to identify **interpretable, closed-form kinetic models** from time-series data, consistent with established kinetic theory.

The repository includes:

- the code used to generate the datasets
- the code used to train the ML-RKIM models
- the datasets used in the paper in CSV format
- a script for reproducing the main figures

---

## Repository Contents

```text

├── README.md
├── ML_KIRM.py
├── ML_KIRM_SEEDS.py
├── ML_KIRM_Ablation.py
├── dataset_prep.py
├── figures.py
├── data/
│   ├── data_poly_isocuring.csv
│   ├── data_CB_cryst_90.csv
│   ├── data_eggyolk_SV.csv
│   ├── data_eggalbumen_SV.csv
│   ├── data_PEEK_253.csv
│   ├── data_prepreg200.csv
│   └── data_AvocadoPuree.csv
├── Results/
│   ├── prediced_data_poly_isocuring.csv
│   ├── prediced_data_CB_cryst_90.csv
│   ├── prediced_data_eggyolk_SV.csv
│   ├── prediced_data_eggalbumen_SV.csv
│   ├── prediced_data_PEEK.csv
│   ├── prediced_data_unfiltered_prepreg200.csv
│   ├── prediced_data_prepreg200.csv
│   └── prediced_data_AvocadoPuree.csv
```

---

## Methodology

ML-RKIM is designed to bridge data-driven modeling with physically meaningful kinetic representations. The framework consists of three primary steps:

1. Construction of a neural network with **physics-informed activation functions** representing candidate kinetic mechanisms  
2. Training of the model using experimental or synthetic time-series data  
3. Application of **sparse regression** to identify the governing terms in the kinetic expression  

The approach builds on the **Sparse Identification of Nonlinear Dynamics (SINDy)** framework, where the system dynamics are expressed as:

$$
\frac{dx(t)}{dt} = f(x(t))
$$

and approximated as a sparse combination of candidate functions:

$$
f(x(t)) \approx \Theta(x(t)) \, \Xi
$$

Here, $\Theta(x)$ represents a library of candidate nonlinear functions and $\Xi$ is a sparse coefficient vector.

---

## Included Codes

### 1. `dataset_prep.py`
This script generates the kinetic datasets used as inputs for model training and testing. It saves the datasets as CSV files in the `data/` directory.

The script currently supports the following materials:

| ID | Material / Process | Output File |
|----|--------------------|-------------|
| `1` | Isothermal Curing of a Thermoset Polymer  | `data_poly_isocuring.csv` |
| `2` | Dynamic Curing of a Thermoset Polymer` | `data_poly_dyncuring.csv` |
| `3` | Isothermal crystallization of Cocoa Butter| `data_CB_cryst_90.csv` |
| `4` | Egg Yolk Gelation - Sous Vide | `data_eggyolk_SV.csv` |
| `5` | Egg Albumen Gelation - Sous Vide | `data_eggalbumen_SV.csv` |
| `6` | Isothermal Crystallization of a Thermoplastic Polymer (PEEK) | `data_PEEK_253.csv` |
| `7` | Avocado Puree Browning | `data_AvocadoPuree.csv` |
| `8` | Isothermal Curing of an Epoxy Prepreg | `data_prepreg200.csv` |

Each generated dataset contains:

- `t`: time
- `T`: temperature
- `x`: extent / state variable
- `xdot`: rate
- `idx`: identifier associated with the processing condition

---

### 2. `ML_KIRM.py`
This is the main training script used for kinetic model discovery.

The script:
- loads a selected dataset from CSV
- trains candidate ML-RKIM models using physics-informed activation functions
- applies LASSO regularization for sparse term selection
- selects the final model based on predictive performance and sparsity
- predicts **kinetic rate behavior** (`xdot`)
- saves the predicted rate results to the `Results/` directory

The learned model has the form:

$$
\frac{dx}{dt} = \sum_{i=1}^{n} \omega_i f_i(x)
$$

where:
- $f_i(x)$ are candidate kinetic mechanisms
- $\omega_i$ are learned coefficients

---

### 3. `figures.py`
This script can reproduce the main figures reported in the paper.

Uses include:
- plotting the original and predicted rate curves
- generating comparison figures for selected materials
- reproducing publication-ready figures from the saved CSV outputs in `Results/`
---

## Kinetic Feature Library

The ML-RKIM model uses a library of candidate kinetic functions derived from kinetic theory:

| Mechanism | Functional Form |
|----------|----------------|
| Nth-order reaction | $(1 - x)^n$ |
| Nucleation-controlled growth | $x^m$ |
| Autocatalytic reaction | $x^m (1 - x)^n$ |
| Diffusion-controlled reaction | $((1 + x)^n - 1)^p$ |
| Empirical growth model | $x^m (1 - x)^n (1 + x)^p$ |
| Avrami model | $k n (1 - x) [-\ln(1 - x)]^{(n-1)/n}$ |
| Extended power law | $k n x [\ln(x)/k]^{(n-1)/n}$ |

These functions are embedded as trainable physics-informed activation functions in the neural network.

---

## Datasets Used in the Paper

The CSV files included in the `data/` folder are the datasets used in the paper:

Synthetic:
- `data_poly_isocuring.csv`
- `data_poly_dyncuring.csv`
- `data_CB_cryst_90.csv`
- `data_eggyolk_SV.csv`
- `data_eggalbumen_SV.csv`
- `data_PEEK_253.csv`
- `data_AvocadoPuree.csv`

Experimental:
- `data_prepreg150.csv`
- `data_prepreg160.csv`
- `data_prepreg170.csv`
- `data_prepreg180.csv`
- `data_prepreg190.csv`
- `data_prepreg200.csv`

Synthetic datasets can be used directly with `ML_KIRM.py` without regeneration, or they can be regenerated with `dataset_prep.py`.

---

## How to Run the Code

### 1. Install dependencies

Create or activate your Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

---

### 2. Train ML-RKIM on a dataset

Open `ML_KIRM.py` and set:

```python
material = 'eggalbumen_SV'
```

or the desired dataset.

Make sure the corresponding CSV file exists in the `data/` directory.

Then run:

```bash
python ML_KIRM.py
```

The script will:
- train candidate models
- print the selected kinetic expression
- save the predicted rate-space output to:

```text
Results/predicted_data_<material>_MLKIRM.csv
```

---

### 3. Reproduce figures

Once the training outputs are available, run:

```bash
python figures.py
```

This script should regenerate the main figures from the paper using the saved results and/or raw datasets.

---

## Output Files

### Training output
The main training script saves a CSV file containing:

- `t`
- `x_original`
- `xdot_original`
- `xdot_pred`

Example:

```text
Results/predicted_data_eggalbumen_SV_MLKIRM.csv
```

### Console output
The training script also reports:
- the selected symbolic kinetic expression
- MSE
- RMSE
- MAE
- R²
- number of functions in the model
- number of active terms after LASSO
- selected hyperparameters

---

## Model Selection Strategy

Sparse regression through LASSO remains the primary sparsity-promoting mechanism in the ML-RKIM framework.

In the current implementation:
- candidate models with 1, 2, and 3 functions are explored
- LASSO determines which terms remain active
- the final model is selected by favoring the **fewest active terms** that meet the specified fit thresholds

---

## Notes

- The model predicts **kinetic rate behavior** (`xdot`) directly from `x`.
- Function combinations are randomly generated for each run, so results may vary.
- Model selection is currently performed using the held-out test set, which is appropriate for exploratory studies but may introduce optimistic bias in formal benchmarking.

---

## Requirements

The required Python libraries are:

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn

---


## Authors

- **Navid Zobeiry** — University of Washington  
- **Paulina Portales** — University of Washington  

---
