#####################################################################
# Date:         May 2026
# Developer:    Navid Zobeiry, navidz@uw.edu      
# Contributor:  Paulina Portales, ppicazo@uw.edu
# Institution:  University of Washington, Seattle, WA
#####################################################################

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.regularizers import l1
from tensorflow.keras.constraints import NonNeg, Constraint


E = False
material = 'PEEK_253'

SEEDS = [23,35,36,43]

df = pd.read_csv("data/data_" + str(material) + ".csv")
time = df['t'].values
x_data = df[['x']].values
xdot_data = df[['xdot']].values

epochs = 200
active_threshold = 1e-5
r2_accept_threshold = 0.999
rmse_accept_threshold = 5e-4

class RangeConstraint(Constraint):
    def __init__(self, min_value=-1, max_value=10):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

class CustomActivationLayer(tf.keras.layers.Layer):
    def __init__(self, function_type="default", seed=None, **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.function_type = function_type
        self.seed = seed

        if function_type in ["default", "m_power", "scaled_poly"]:
            self.m = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=3.0, seed=seed),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.m = None

        if function_type in ["default", "n_power", "custom", "scaled_poly", "avrami", "ext_powerlaw"]:
            self.n = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=3, seed=seed),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.n = None

        if function_type in ["custom", "scaled_poly"]:
            self.p = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=3, seed=seed),
                constraint=RangeConstraint(min_value=0, max_value=5),
                trainable=True
            )
        else:
            self.p = None

        if function_type in ["avrami", "ext_powerlaw"]:
            if function_type == "avrami":
                init_min, init_max = 0.00001, 10.0
                constraint = RangeConstraint(min_value=0.00001, max_value=10.0)
            else:
                init_min, init_max = -10.0, -0.00001
                constraint = RangeConstraint(min_value=-10.0, max_value=-0.00001)

            self.k = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.RandomUniform(minval=init_min, maxval=init_max, seed=seed),
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

def generate_function_combinations(n_functions, seed):
    available_function_types = ["default", "m_power", "n_power", "custom", 'scaled_poly', 'avrami', 'ext_powerlaw']
    rng = np.random.default_rng(seed + n_functions)

    function_combinations = []
    for _ in range(5):
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
    seed,
    n_functions=3,
    epochs=epochs
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    inputs = Input(shape=(1,))
    activated_layers = [
        CustomActivationLayer(function_type=function_types[i], seed=seed)(inputs)
        for i in range(n_functions)
    ]

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
        shuffle=False
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
                term = f"{weight[0]: .4f} * x^{m:.4f} * (1 - x)^{n:.4f} * (1 + x)^{p:.4f}"
            elif function_types[i] == "avrami":
                term = f"{weight[0]:.4f} * {k:.4f} * {n:.4f} * (1 - x) * (-ln(1 - x))^(({n - 1:.4f}) / {n:.4f})"
            elif function_types[i] == "ext_powerlaw":
                term = f"{weight[0]:.4f} * {k:.4f} * {n:.4f} * x * (ln(x)/{k:.4f})^(({n - 1:.4f})/{n:.4f})"

            terms.append(term)

    function_expression = " + ".join(terms)

    return model, history, mse, rmse, mae, r2, function_expression, dense_weights, n_active_terms


lambda_values = [1e-17, 1e-15, 1e-10]
learning_rates = [0.01, 0.001]
batch_sizes = [8, 10, 16]
n_functions_values = [1, 2, 3]

seed_results = []

for seed in SEEDS:
    print(f"\n==============================")
    print(f"Running seed: {seed}")
    print(f"==============================")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
        x_data, xdot_data, time, test_size=0.2, random_state=seed
    )

    best_overall_result = None
    all_results = []

    for lasso_lambda in lambda_values:
        for learning_rate_adam in learning_rates:
            for batch_size in batch_sizes:
                for n_functions in n_functions_values:
                    function_combinations = generate_function_combinations(n_functions, seed)

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
                                seed,
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
                            print(f"Skipping model due to error: {e}")
                            continue

    if all_results:
        best_overall_result = None

        for target_active_terms in [1, 2, 3]:
            candidates = [
                result for result in all_results
                if result[13] == target_active_terms
            ]

            acceptable = [
                result for result in candidates
                if (result[4] >= r2_accept_threshold) and (result[2] <= rmse_accept_threshold)
            ]

            if acceptable:
                best_overall_result = min(
                    acceptable,
                    key=lambda x: (x[1], x[12])
                )
                break

        if best_overall_result is None:
            best_overall_result = min(
                all_results,
                key=lambda x: (x[13], x[1], x[12])
            )

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
        ) = best_overall_result

        print(f"\nBest Model for seed {seed}:")
        print(f"  Expression: {function_expression}")
        print(f"  MSE: {mse:.6e}, RMSE: {rmse:.6e}, MAE: {mae:.6e}, R²: {r2:.6f}")
        print(f"  Functions in model: {best_n_functions}")
        print(f"  Active terms after LASSO: {best_n_active_terms}")
        print(f"  Lambda: {best_lambda}, Learning rate: {best_lr}, Batch size: {best_batch_size}")

        seed_results.append({
            "seed": seed,
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
            "function_types": function_types
        })

        sort_idx = np.argsort(time)

        t_all = time[sort_idx]
        x_all = x_data[sort_idx]
        xdot_all = xdot_data[sort_idx]

        y_pred_all = best_model.predict(x_all, verbose=0)

        df_pred_seed = pd.DataFrame({
            "t": t_all,
            "x_original": x_all.flatten(),
            "xdot_original": xdot_all.flatten(),
            "xdot_pred": y_pred_all.flatten()
        })

        pred_output_path = (
            f"Results/predicted_data_{material}_seed_{seed}.csv"
        )

        df_pred_seed.to_csv(pred_output_path, index=False)

        print(
            f"Predicted dataset for seed {seed} saved to: "
            f"{pred_output_path}"
        )

    else:
        print(f"No models were successfully trained for seed {seed}.")

seed_results_df = pd.DataFrame(seed_results)

output_path = f"Results/seed_reproducibility_{material}.csv"
seed_results_df.to_csv(output_path, index=False)

print(f"\nSeed reproducibility results saved to: {output_path}")

print("\nSummary:")
print(seed_results_df[["seed", "r2", "rmse", "n_active_terms", "expression"]])

print("\nR² mean:", seed_results_df["r2"].mean())
print("R² std:", seed_results_df["r2"].std())
print("RMSE mean:", seed_results_df["rmse"].mean())
print("RMSE std:", seed_results_df["rmse"].std())