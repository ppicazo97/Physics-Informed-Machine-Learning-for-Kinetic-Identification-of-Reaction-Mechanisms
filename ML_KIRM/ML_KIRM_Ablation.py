#####################################################################
# Date:         May 2026
# Developer:    Navid Zobeiry, navidz@uw.edu
# Contributor:  Paulina Portales, ppicazo@uw.edu
# Institution:  University of Washington, Seattle, WA
#####################################################################

import os
import random

############################
# Settings
############################

SEED = 36
material = 'PEEK_253'

USE_REPRODUCIBLE_SEEDS = True
SUPPRESS_TF_MESSAGES = True

epochs = 150
active_threshold = 1e-5
r2_accept_threshold = 0.999
rmse_accept_threshold = 5e-4

lambda_values = [1e-17, 1e-15, 1e-10]
learning_rates = [0.01, 0.001]
batch_sizes = [8, 10, 16]
n_functions_values = [1, 2, 3]

n_function_combinations = 5

BASE_FUNCTION_TYPES = [
    "default",
    "m_power",
    "n_power",
    "custom",
    "scaled_poly",
    "avrami",
    "ext_powerlaw"
]

# "interpretable" = avoids blank expressions and chooses simplest accurate model.
# "lowest_mse" = closest to classic lowest-error selection.
SELECTION_MODE = "interpretable"

RUN_SCENARIOS = {
    "baseline": True,
    "noise": {
        "enabled": True,
        "levels": [0.01, 0.05, 0.10]
    },
    "data_fraction": {
        "enabled": True,
        "fractions":[0.25, 0.50, 0.75, 1.00]
    },
    "sampling_density": {
        "enabled": True,
        "steps": [2, 5, 10]
    },
    "missing_middle": {
        "enabled": True,
        "fractions": [0.30]
    },
    "library_variation": {
        "enabled": True,
        "excluded_function_sets": [
            ["avrami"]
        ]
    }
}

if SUPPRESS_TF_MESSAGES:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if USE_REPRODUCIBLE_SEEDS:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

if SUPPRESS_TF_MESSAGES:
    tf.get_logger().setLevel("ERROR")

if USE_REPRODUCIBLE_SEEDS:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.regularizers import l1
from tensorflow.keras.constraints import NonNeg, Constraint

############################
# LOAD DATA
############################

E = False

df = pd.read_csv("data/data_" + str(material) + ".csv")
time = df['t'].values
x_data = df[['x']].values
xdot_data = df[['xdot']].values

############################
# MODELS 
############################

class RangeConstraint(Constraint):
    def __init__(self, min_value=-1, max_value=10):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)


class CustomActivationLayer(tf.keras.layers.Layer):
    def __init__(self, function_type="default", **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.function_type = function_type

        if function_type in ["default", "m_power", "scaled_poly"]:
            self.m = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=0,
                    maxval=3.0,
                    seed=SEED if USE_REPRODUCIBLE_SEEDS else None
                ),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.m = None

        if function_type in ["default", "n_power", "custom", "scaled_poly", "avrami", "ext_powerlaw"]:
            self.n = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=0,
                    maxval=3,
                    seed=SEED if USE_REPRODUCIBLE_SEEDS else None
                ),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.n = None

        if function_type in ["custom", "scaled_poly"]:
            self.p = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=0,
                    maxval=3,
                    seed=SEED if USE_REPRODUCIBLE_SEEDS else None
                ),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.p = None

        if function_type in ["avrami", "ext_powerlaw"]:
            if function_type == "avrami":
                init_min, init_max = 0.00001, 10.0
                constraint = RangeConstraint(min_value=0.00001, max_value=10.0)
            elif function_type == "ext_powerlaw":
                init_min, init_max = -10.0, -0.00001
                constraint = RangeConstraint(min_value=-10.0, max_value=-0.00001)

            self.k = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=init_min,
                    maxval=init_max,
                    seed=SEED if USE_REPRODUCIBLE_SEEDS else None
                ),
                constraint=constraint,
                trainable=True
            )
        else:
            self.k = None

    def call(self, inputs):
        x = inputs
        x_clipped = tf.clip_by_value(x, 1e-6, 1 - 1e-6)

        if self.function_type == "default":
            return tf.pow(x_clipped, self.m) * tf.pow(1.0 - x_clipped, self.n)
        elif self.function_type == "m_power":
            return tf.pow(x_clipped, self.m)
        elif self.function_type == "n_power":
            return tf.pow(1.0 - x_clipped, self.n)
        elif self.function_type == "custom":
            return tf.pow(tf.pow(1.0 + x_clipped, self.n) - 1.0, self.p)
        elif self.function_type == 'scaled_poly':
            return tf.pow(x_clipped, self.m) * tf.pow(1 - x_clipped, self.n) * tf.pow(1 + x_clipped, self.p)
        elif self.function_type == "avrami":
            log_term = -tf.math.log(1.0 - x_clipped)
            inner = tf.pow(log_term, (self.n - 1.0) / self.n)
            return self.k * self.n * inner * (1.0 - x_clipped)
        elif self.function_type == "ext_powerlaw":
            log_term = tf.math.log(x_clipped)
            inner = tf.pow(log_term / self.k, (self.n - 1.0) / self.n)
            return self.k * self.n * x_clipped * inner
        else:
            raise ValueError("Invalid function type")


def generate_function_combinations(n_functions, available_function_types=None):
    if available_function_types is None:
        available_function_types = BASE_FUNCTION_TYPES

    rng = np.random.default_rng(SEED + n_functions) if USE_REPRODUCIBLE_SEEDS else np.random.default_rng()

    function_combinations = []
    for _ in range(n_function_combinations):
        combination = rng.choice(available_function_types, size=n_functions, replace=True)
        function_combinations.append(list(combination))

    return function_combinations


def build_and_train_model(
    x_train,
    y_train,
    x_test,
    y_test,
    function_types,
    lasso_lambda,
    learning_rate_adam,
    batch_size,
    n_functions=3,
    epochs=epochs
):
    tf.keras.backend.clear_session()

    if USE_REPRODUCIBLE_SEEDS:
        tf.keras.utils.set_random_seed(SEED)

    inputs = Input(shape=(1,))
    activated_layers = [CustomActivationLayer(function_type=function_types[i])(inputs) for i in range(n_functions)]
    combined_layer = concatenate(activated_layers)

    output = Dense(
        1,
        activation='linear',
        use_bias=False,
        kernel_constraint=NonNeg(),
        kernel_regularizer=l1(lasso_lambda)
    )(combined_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_adam),
        loss='mean_squared_error'
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
        shuffle=False if USE_REPRODUCIBLE_SEEDS else True
    )

    y_pred = model.predict(x_test, verbose=0)

    m_values, n_values, p_values, k_values = [], [], [], []
    for layer in model.layers:
        if isinstance(layer, CustomActivationLayer):
            m_values.append(layer.m.numpy().item() if layer.m is not None else None)
            n_values.append(layer.n.numpy().item() if layer.n is not None else None)
            p_values.append(layer.p.numpy().item() if layer.p is not None else None)
            k_values.append(layer.k.numpy().item() if layer.k is not None else None)

    dense_weights = model.layers[-1].get_weights()[0]
    n_active_terms = int(np.sum(np.abs(dense_weights) > active_threshold))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    terms = []
    for i, (m, n, p, k, weight) in enumerate(zip(m_values, n_values, p_values, k_values, dense_weights)):
        if np.abs(weight) > active_threshold:
            if function_types[i] == "default":
                term = f"{weight[0]:.4f} * x^{m:.4f} * (1 - x)^{n:.4f}"
            elif function_types[i] == "m_power":
                term = f"{weight[0]:.4f} * x^{m:.4f}"
            elif function_types[i] == "n_power":
                term = f"{weight[0]:.4f} * (1 - x)^{n:.4f}"
            elif function_types[i] == "custom":
                term = f"{weight[0]:.4f} * ((1 + x)^{n:.4f} - 1)^{p:.4f}"
            elif function_types[i] == 'scaled_poly':
                term = f"{weight[0]:.4f} * x^{m:.4f} * (1 - x)^{n:.4f} * (1 + x)^{p:.4f}"
            elif function_types[i] == "avrami":
                term = f"{weight[0]:.4f} * {k:.4f} * {n:.4f} * (1 - x) * (-ln(1 - x))^(({n - 1:.4f}) / {n:.4f})"
            elif function_types[i] == "ext_powerlaw":
                term = f"{weight[0]:.4f} * {k:.4f} * {n:.4f} * x * (ln(x)/{k:.4f})^(({n - 1:.4f})/{n:.4f})"
            terms.append(term)

    function_expression = " + ".join(terms)

    return model, history, mse, rmse, mae, r2, function_expression, dense_weights, n_active_terms

############################
# SELECTION AND EXPERIMENT 
############################

def select_best_model(all_results):
    if SELECTION_MODE == "lowest_mse":
        best_result = min(all_results, key=lambda x: x[1])
        selection_rule = "lowest MSE model"

    elif SELECTION_MODE == "interpretable":
        nonblank_results = [
            result for result in all_results
            if result[13] > 0 and result[7] != ""
        ]

        good_results = [
            result for result in nonblank_results
            if (result[4] >= r2_accept_threshold)
            and (result[2] <= rmse_accept_threshold)
        ]

        if good_results:
            best_result = min(good_results, key=lambda x: (x[13], x[1], x[12]))
            selection_rule = "simplest nonblank model meeting accuracy thresholds"

        elif nonblank_results:
            best_result = min(nonblank_results, key=lambda x: x[1])
            selection_rule = "best nonblank model because no model met thresholds"

        else:
            best_result = min(all_results, key=lambda x: x[1])
            selection_rule = "best MSE model, but expression may be blank"

    else:
        raise ValueError("SELECTION_MODE must be either 'interpretable' or 'lowest_mse'")

    return best_result, selection_rule


def run_ml_kirm_experiment(
    x_data_input,
    xdot_data_input,
    time_input,
    experiment_name,
    available_function_types=None
):
    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
        x_data_input,
        xdot_data_input,
        time_input,
        test_size=0.2,
        random_state=SEED if USE_REPRODUCIBLE_SEEDS else None
    )

    all_results = []

    for lasso_lambda in lambda_values:
        for learning_rate_adam in learning_rates:
            for batch_size in batch_sizes:
                for n_functions in n_functions_values:
                    function_combinations = generate_function_combinations(
                        n_functions,
                        available_function_types=available_function_types
                    )

                    for function_types in function_combinations:
                        try:
                            model, history, mse, rmse, mae, r2, function_expression, dense_weights, n_active_terms = build_and_train_model(
                                x_train,
                                y_train,
                                x_test,
                                y_test,
                                function_types,
                                lasso_lambda,
                                learning_rate_adam,
                                batch_size,
                                n_functions=n_functions,
                                epochs=epochs
                            )

                            all_results.append((
                                function_types,
                                mse,
                                rmse,
                                mae,
                                r2,
                                model,
                                history,
                                function_expression,
                                dense_weights,
                                lasso_lambda,
                                learning_rate_adam,
                                batch_size,
                                n_functions,
                                n_active_terms
                            ))

                        except Exception as e:
                            print(f"[{experiment_name}] Skipping model due to error: {e}")
                            continue

    if not all_results:
        print(f"\n[{experiment_name}] No models were successfully trained.")
        return None

    best_result, selection_rule = select_best_model(all_results)

    (
        function_types,
        mse,
        rmse,
        mae,
        r2,
        best_model,
        best_history,
        function_expression,
        best_dense_weights,
        best_lambda,
        best_lr,
        best_batch_size,
        best_n_functions,
        best_n_active_terms
    ) = best_result

    print(f"\nExperiment: {experiment_name}")
    print(f"  Expression: {function_expression}")
    print(f"  MSE: {mse:.6e}, RMSE: {rmse:.6e}, MAE: {mae:.6e}, R²: {r2:.6f}")
    print(f"  Functions in model: {best_n_functions}")
    print(f"  Active terms after LASSO: {best_n_active_terms}")
    print(f"  Lambda: {best_lambda}, Learning rate: {best_lr}, Batch size: {best_batch_size}")
    print(f"  Selection rule: {selection_rule}")

    return {
        "experiment": experiment_name,
        "expression": function_expression,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_functions": best_n_functions,
        "n_active_terms": best_n_active_terms,
        "lambda": best_lambda,
        "learning_rate": best_lr,
        "batch_size": best_batch_size,
        "selection_rule": selection_rule,
        "function_types": function_types,
        "model": best_model,
        "history": best_history
    }


def add_result(result, summary_results):
    if result is not None:
        summary_results.append(result)

############################
# RUN SELECTED SCENARIOS
############################

os.makedirs("Results", exist_ok=True)
summary_results = []

baseline_result = None

if RUN_SCENARIOS.get("baseline", False):
    baseline_result = run_ml_kirm_experiment(
        x_data,
        xdot_data,
        time,
        experiment_name="baseline",
        available_function_types=BASE_FUNCTION_TYPES
    )
    add_result(baseline_result, summary_results)

if RUN_SCENARIOS.get("noise", {}).get("enabled", False):
    for noise_level in RUN_SCENARIOS["noise"]["levels"]:
        rng = np.random.default_rng(SEED)
        noise = noise_level * np.std(xdot_data) * rng.normal(size=xdot_data.shape)
        xdot_noisy = xdot_data + noise

        result = run_ml_kirm_experiment(
            x_data,
            xdot_noisy,
            time,
            experiment_name=f"noise_{int(noise_level * 100)}pct",
            available_function_types=BASE_FUNCTION_TYPES
        )
        add_result(result, summary_results)

if RUN_SCENARIOS.get("data_fraction", {}).get("enabled", False):
    for fraction in RUN_SCENARIOS["data_fraction"]["fractions"]:
        rng = np.random.default_rng(SEED)
        n_points = int(len(x_data) * fraction)

        selected_idx = rng.choice(
            len(x_data),
            size=n_points,
            replace=False
        )
        selected_idx = np.sort(selected_idx)

        result = run_ml_kirm_experiment(
            x_data[selected_idx],
            xdot_data[selected_idx],
            time[selected_idx],
            experiment_name=f"data_fraction_{int(fraction * 100)}pct",
            available_function_types=BASE_FUNCTION_TYPES
        )
        add_result(result, summary_results)

if RUN_SCENARIOS.get("sampling_density", {}).get("enabled", False):
    time_sorted_idx = np.argsort(time)

    for step in RUN_SCENARIOS["sampling_density"]["steps"]:
        selected_idx = time_sorted_idx[::step]

        result = run_ml_kirm_experiment(
            x_data[selected_idx],
            xdot_data[selected_idx],
            time[selected_idx],
            experiment_name=f"sampling_every_{step}th_point",
            available_function_types=BASE_FUNCTION_TYPES
        )
        add_result(result, summary_results)

if RUN_SCENARIOS.get("missing_middle", {}).get("enabled", False):
    t_min = np.min(time)
    t_max = np.max(time)
    t_range = t_max - t_min

    for missing_fraction in RUN_SCENARIOS["missing_middle"]["fractions"]:
        missing_start = t_min + ((1.0 - missing_fraction) / 2.0) * t_range
        missing_end = t_max - ((1.0 - missing_fraction) / 2.0) * t_range

        mask = (time < missing_start) | (time > missing_end)

        result = run_ml_kirm_experiment(
            x_data[mask],
            xdot_data[mask],
            time[mask],
            experiment_name=f"missing_middle_{int(missing_fraction * 100)}pct_time",
            available_function_types=BASE_FUNCTION_TYPES
        )
        add_result(result, summary_results)

if RUN_SCENARIOS.get("library_variation", {}).get("enabled", False):
    for excluded_functions in RUN_SCENARIOS["library_variation"]["excluded_function_sets"]:
        available_functions = [
            f for f in BASE_FUNCTION_TYPES
            if f not in excluded_functions
        ]

        excluded_label = "_".join(excluded_functions)

        result = run_ml_kirm_experiment(
            x_data,
            xdot_data,
            time,
            experiment_name=f"library_without_{excluded_label}",
            available_function_types=available_functions
        )
        add_result(result, summary_results)

############################
# SAVE COMPARISON TABLE
############################

summary_df = pd.DataFrame([
    {
        "experiment": r["experiment"],
        "expression": r["expression"],
        "mse": r["mse"],
        "rmse": r["rmse"],
        "mae": r["mae"],
        "r2": r["r2"],
        "n_functions": r["n_functions"],
        "n_active_terms": r["n_active_terms"],
        "lambda": r["lambda"],
        "learning_rate": r["learning_rate"],
        "batch_size": r["batch_size"],
        "selection_rule": r["selection_rule"],
        "function_types": r["function_types"]
    }
    for r in summary_results
])

summary_path = f"Results/comparison_summary_{material}_SEED: {SEED}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nComparison summary saved to: {summary_path}")

############################
# BASELINE
############################

if baseline_result is not None:
    best_model = baseline_result["model"]
    best_history = baseline_result["history"]

    plt.figure(figsize=(10, 6))
    plt.plot(best_history.history['loss'], label='Training loss')
    plt.plot(best_history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss - Baseline')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    time_sorted_idx = np.argsort(time)
    t_all = time[time_sorted_idx]
    x_all = x_data[time_sorted_idx]
    xdot_all = xdot_data[time_sorted_idx]

    y_pred_all = best_model.predict(x_all, verbose=0)

    plt.figure(figsize=(10, 5))
    plt.plot(t_all, xdot_all, '.', ms=3, alpha=0.5, label='Original rate (xdot)')
    plt.plot(t_all, y_pred_all, '-', lw=1.5, label='Predicted rate (model)')
    plt.xlabel('Time')
    plt.ylabel('xdot')
    plt.title('Original vs Predicted xdot - Baseline')
    plt.legend()
    plt.tight_layout()
    plt.show()

    x_sim = np.zeros_like(x_all, dtype=float)
    x_sim[0, 0] = float(x_all[0, 0])

    for i in range(1, len(t_all)):
        dt = t_all[i] - t_all[i - 1]
        x_prev = np.array([[x_sim[i - 1, 0]]], dtype=float)
        xdot_i = best_model(x_prev, training=False).numpy()[0, 0]
        x_sim[i, 0] = x_sim[i - 1, 0] + xdot_i * dt
        x_sim[i, 0] = np.clip(x_sim[i, 0], 1e-6, 1 - 1e-6)

    df_pred = pd.DataFrame({
        "t": t_all,
        "x_original": x_all.flatten(),
        "xdot_original": xdot_all.flatten(),
        "xdot_pred": y_pred_all.flatten(),
        "x_pred_integrated": x_sim.flatten()
    })

    output_path = f"Results/predicted_data_{material}_MLKIRM_baseline.csv"
    df_pred.to_csv(output_path, index=False)
    print(f"Predicted baseline dataset saved to: {output_path}")

    r2_rate = r2_score(xdot_all, y_pred_all)
    rmse_rate = mean_squared_error(xdot_all, y_pred_all, squared=False)
    mae_rate = mean_absolute_error(xdot_all, y_pred_all)

    print(f"Baseline full-series rate metrics  R²={r2_rate:.4f}  RMSE={rmse_rate:.4e}  MAE={mae_rate:.4e}")

else:
    print("\nBaseline was not run, so no baseline plots/prediction file were generated.")
