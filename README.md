# ML-KIRM: Machine Learning Framework for Interpretable Kinetic Model Discovery

This repository implements **ML-KIRM**, a physics-informed machine learning framework developed to identify **interpretable, closed-form kinetic models** from time-series data, consistent with established kinetic theory.

The framework learns governing equations of the form:

$$
\frac{dx}{dt} = f(x)
$$

where $x$ represents the extent (or state variable) and $f(x)$ captures the underlying reaction mechanisms.

---

## Methodology

ML-KIRM is designed to bridge data-driven modeling with physically meaningful kinetic representations. The framework consists of three primary steps:

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

## Model Architecture

The ML-KIRM framework is implemented as a neural network that embeds kinetic theory directly into its structure.

The architecture consists of:

- **Input layer**: reaction extent $x$  
- **Custom activation layer**: domain-informed kinetic functions  
- **Output layer**: sparse linear combination of activated functions  

Each activation corresponds to a candidate kinetic mechanism $f_i(x)$, parameterized by trainable coefficients (e.g., $m$, $n$, $p$), constrained within physically meaningful ranges.

The resulting model takes the form:

$$
\frac{dx}{dt} = \sum_{i=1}^{n} \omega_i f_i(x)
$$

where:
- $f_i(x)$ are candidate mechanisms from the kinetic feature library  
- $\omega_i$ are learned coefficients  

---

## Kinetic Feature Library

The model includes a library of candidate functions derived from kinetic theory, representing common reaction behaviors:

| Mechanism | Functional Form |
|----------|----------------|
| Nth-order reaction | $(1 - x)^n$ |
| Nucleation-controlled growth | $x^m$ |
| Autocatalytic reaction | $x^m (1 - x)^n$ |
| Diffusion-controlled reaction | $((1 + x)^n - 1)^p$ |
| Empirical growth model | $x^m (1 - x)^n (1 + x)^p$ |
| Avrami model | $k n (1 - x) [-\ln(1 - x)]^{(n-1)/n}$ |
| Extended power law | $k n x [\ln(x)/k]^{(n-1)/n}$ |

These functions capture diverse kinetic regimes, including:
- deceleratory behavior  
- sigmoidal kinetics  
- nucleation and growth processes  
- diffusion-limited regimes  

The framework allows flexible combinations of these functions to model **multistage and overlapping kinetics**.

---

## Sparse Regression

To ensure interpretability, sparsity is enforced using **LASSO regularization**, which promotes selection of a minimal subset of active mechanisms.

The regularization term is defined as:

$$
\phi(w) = \|w\|_1
$$

This drives non-essential coefficients toward zero, yielding compact and interpretable kinetic expressions.

---

## Training and Optimization

Model training is performed using time-series datasets containing:

- reaction extent ($x$)
- time ($t$)
- reaction rate ($\dot{x}$)

A grid search is used to explore hyperparameters, including:

- LASSO regularization strength ($\lambda$)
- learning rate
- batch size
- number of active functions

Each model is trained using **random combinations of kinetic functions**, enabling exploration of diverse candidate mechanisms.

---

## Quick Start

1. Place your dataset in:
```
data_various/data_<material>.csv
```

2. Open the script and select a dataset:

```python
material = "1"
```

Available examples include:
- `"1"` — Iso Curing SC1008  
- `"avrami_90min_2"` — Cacao butter crystallization  
- `"eggalbumen_SV"` — Egg gelation  
- `"PEEK_253"` — PEEK crystallization  
- `"AvocadoPuree"` — Enzymatic browning  

3. Run the script:

```bash
python your_script.py
```

4. Outputs will be saved to:

```
data/predicted_data_<material>_3func.csv
```

---

## Workflow

1. Load dataset $(x, \dot{x}, t)$  
2. Split into training and test sets  
3. Generate random combinations of kinetic functions  
4. Train models across hyperparameter configurations  
5. Select best-performing model (based on test MSE)  
6. Extract symbolic kinetic expression  
7. Predict $\dot{x}$ over the full dataset  
8. Integrate predicted rates to reconstruct $x(t)$  
9. Evaluate performance and save outputs  

---

## Output

### Saved Data

```
data/predicted_data_<material>_3func.csv
```

Includes:
- original measurements
- predicted rates
- integrated state evolution

---

### Model Evaluation

Performance is evaluated in two domains:

- **Rate space**: predicted vs. observed $\dot{x}$  
- **State space**: reconstructed $x(t)$  

Metrics reported:
- MSE, RMSE, MAE, R²  

---

## Notes

- Model selection is performed using the test set.  
  This is suitable for exploratory analysis but may introduce optimistic bias in formal benchmarking.

- Function combinations are randomly generated for each run (no fixed seed).

- The model enforces:
  - non-negative coefficients  
  - bounded parameters  
  - domain clipping $x \in (0,1)$ for numerical stability  

---

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- scikit-learn

Install with:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

---

## Authors

- **Navid Zobeiry** — University of Washington  
- **Paulina Portales** — University of Washington  

---

## Acknowledgments

This work was developed for research in **data-driven kinetic modeling** and **equation discovery in materials systems**, with applications to polymer curing, crystallization, gelation, and enzymatic processes.
