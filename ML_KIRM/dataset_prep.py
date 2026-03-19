#####################################################################
# Date:         March 2026
# Developer:    Navid Zobeiry, navidz@uw.edu      
# Contributor:  Paulina Portales, ppicazo@uw.edu
# Institution:  University of Washington, Seattle, WA
#####################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Parameters ###

mat = 1

T_ISO = 180
TIMEHOLD = 120
dt = 2
TDOT = 4

def cycle(dt, T_hist):
    times = []
    temperatures = []

    t_i = 0
    T_i = T_hist[0][0]

    times.append(t_i)
    temperatures.append(T_i)

    for i in range(len(T_hist)):
        steps = int(T_hist[i][2] / (dt / 60))
        dt_temp = T_hist[i][2] / steps
        ramp = (T_hist[i][1] - T_hist[i][0]) / T_hist[i][2]

        for _ in range(steps):
            t = t_i + dt_temp
            times.append(t * 60)

            if T_hist[i][0] != T_hist[i][1]:
                T_i = T_i + dt_temp * ramp

            temperatures.append(T_i)
            t_i = t

    return np.array(times), np.array(temperatures)

def save_and_plot(t_train, T_train, x_train, xdot_train, idx, filename):
    df = pd.DataFrame({
        "t": t_train,
        "T": T_train,
        "x": x_train,
        "xdot": xdot_train,
        "idx": idx
    })

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(df.describe()[["idx", "t", "T", "x", "xdot"]])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    axes[0].plot(df["t"], df["x"], label="x")
    axes[0].legend()
    axes[0].set(xlabel="time", ylabel="x")

    axes[1].plot(df["t"], df["xdot"], label="xdot", alpha=0.5)
    axes[1].legend()
    axes[1].set(xlabel="time", ylabel="xdot")

    plt.tight_layout()
    plt.show()

if mat == 1:
    T = T_ISO + 273.15
    A = 1.52e9
    E = 96.1e3
    n = 1.37
    R = 8.314

    T_hist = [[T, T, TIMEHOLD]]
    t_train, T_train = cycle(dt, T_hist)

    def xdot(x, T):
        return A * np.exp(-E / (R * T)) * (1 - x) ** n

    x0 = 0.001
    x_train = [x0]
    xdot_train = [xdot(x0, T)]

    for i in range(1, len(t_train)):
        xdot_val = xdot(x_train[-1], T_train[i - 1])
        xdot_train.append(xdot_val)
        x_train.append(x_train[-1] + xdot_val * dt)

    T_train = np.full_like(t_train, T, dtype=float)
    idx = np.full_like(t_train, T_ISO, dtype=float)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train,
            T_train=T_train,
            x_train=np.array(x_train),
            xdot_train=np.array(xdot_train),
            idx=idx,
            filename="data_various/data_poly_isocuring.csv"
        )

elif mat == 2:
    A = 1.52e9
    E = 96.1e3
    n = 1.37
    R = 8.314

    T_i = 273.15
    T_f = 523.15
    Tdot = TDOT
    time = (T_f - T_i) / Tdot * 60

    dt = 2
    t_train = np.arange(0, time, dt)

    def xdot(x, T):
        return A * np.exp(-E / (R * T)) * (1 - x) ** n

    def temp(t):
        return T_i + Tdot * (t / 60)

    x0 = 0.001
    x_train = [x0]
    T_train = [T_i]
    xdot_train = [0]
    idx = []

    for i in range(1, len(t_train)):
        xdot_val = xdot(x_train[-1], T_train[-1])
        x_train.append(x_train[-1] + xdot_val * dt)
        T_train.append(temp(t_train[i]))
        xdot_train.append(xdot_val)
        idx.append(Tdot)

    idx = np.full(len(t_train), Tdot, dtype=float)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train,
            T_train=np.array(T_train),
            x_train=np.array(x_train),
            xdot_train=np.array(xdot_train),
            idx=idx,
            filename="data_various/data_poly_dyncuring.csv"
        )

elif mat == 3:
    n_avrami = 2.53
    k_avrami = 2.87e-3
    T_ISO = 15
    Tc = T_ISO + 273.15
    TIMEHOLD = 90
    dt = 2

    t_train = np.arange(0, TIMEHOLD * 60 + dt, dt)
    t_min = t_train / 60

    def xdot_avrami(t):
        return n_avrami * k_avrami * (t ** (n_avrami - 1)) * np.exp(-k_avrami * t ** n_avrami)

    x0 = 0.001
    x_train = [x0]
    xdot_train = [xdot_avrami(t_min[0])]
    T_train = [Tc]
    idx = [T_ISO]

    for i in range(1, len(t_train)):
        xdot_val = xdot_avrami(t_min[i])
        xdot_train.append(xdot_val)
        T_train.append(Tc)
        x_train.append(x_train[-1] + xdot_val * (dt / 60))
        idx.append(T_ISO)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train / 60,
            T_train=np.array(T_train),
            x_train=np.array(x_train),
            xdot_train=np.array(xdot_train),
            idx=np.array(idx),
            filename="data_various/data_CB_cryst_90.csv"
        )

elif mat == 4:
    A = 2.72e50
    E = 3.443e5
    n = 1.0
    R = 8.314

    T_ISO = 65
    T = T_ISO + 273.15
    TIMEHOLD = 60
    dt = 2

    T_hist = [[T, T, TIMEHOLD]]
    t_train, T_train = cycle(dt, T_hist)

    def xdot(x, T):
        return A * np.exp(-E / (R * T)) * (1 - x) ** n

    x0 = 0.001
    x_train = [x0]
    xdot_train = [xdot(x0, T)]
    idx = [T_ISO]

    for i in range(1, len(t_train)):
        xdot_val = xdot(x_train[-1], T_train[i - 1])
        x_train.append(x_train[-1] + xdot_val * dt)
        xdot_train.append(xdot_val)
        idx.append(T_ISO)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train,
            T_train=T_train,
            x_train=np.array(x_train),
            xdot_train=np.array(xdot_train),
            idx=np.array(idx),
            filename="data_various/data_eggyolk_SV.csv"
        )

elif mat == 5:
    A = 4.85e60
    E = 4.185e5
    n = 1.0
    R = 8.314

    T_ISO = 65
    T = T_ISO + 273.15
    TIMEHOLD = 60
    dt = 2

    T_hist = [[T, T, TIMEHOLD]]
    t_train, T_train = cycle(dt, T_hist)

    def xdot(x, T):
        return A * np.exp(-E / (R * T)) * (1 - x) ** n

    x0 = 0.001
    x_train = [x0]
    xdot_train = [xdot(x0, T)]
    idx = [T_ISO]

    for i in range(1, len(t_train)):
        xdot_val = xdot(x_train[-1], T_train[i - 1])
        x_train.append(x_train[-1] + xdot_val * dt)
        xdot_train.append(xdot_val)
        idx.append(T_ISO)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train,
            T_train=T_train,
            x_train=np.array(x_train),
            xdot_train=np.array(xdot_train),
            idx=np.array(idx),
            filename="data_various/data_eggalbumen_SV.csv"
        )

elif mat == 6:
    n_avrami = 0.95
    k_avrami = 1.82e-2
    T_ISO = 253
    Tc = T_ISO + 273.15
    TIMEHOLD = 15
    dt = 2

    t_train = np.arange(0, TIMEHOLD * 60 + dt, dt)
    t_min = t_train / 60

    def x_avrami(t):
        return 1 - np.exp(-k_avrami * t ** n_avrami)

    x_train = x_avrami(t_min)

    dx = np.diff(x_train)
    dt_num = np.diff(t_train)
    xdot_train = np.zeros_like(x_train)
    xdot_train[1:] = dx / dt_num
    xdot_train[0] = xdot_train[1]

    T_train = np.full_like(t_train, Tc, dtype=float)
    idx = np.full_like(t_train, T_ISO, dtype=float)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train / 60,
            T_train=T_train,
            x_train=x_train,
            xdot_train=xdot_train,
            idx=idx,
            filename="data_various/data_PEEK_253.csv"
        )

elif mat == 7:
    n = 1.7
    k = 9.3e-5
    T_ISO = 4
    Tc = T_ISO + 273.15
    TIMEHOLD = 750
    dt = 2

    t_train = np.arange(0, TIMEHOLD * 60 + dt, dt)
    t_min = t_train / 60

    def x_browning(t):
        return np.exp(-k * t ** n)

    x_train = x_browning(t_min)

    dx = np.diff(x_train)
    dt_num = np.diff(t_train)
    xdot_train = np.zeros_like(x_train)
    xdot_train[1:] = dx / dt_num
    xdot_train[0] = xdot_train[1]

    T_train = np.full_like(t_train, Tc, dtype=float)
    idx = np.full_like(t_train, T_ISO, dtype=float)

    if __name__ == "__main__":
        save_and_plot(
            t_train=t_train / 60,
            T_train=T_train,
            x_train=x_train,
            xdot_train=xdot_train,
            idx=idx,
            filename="data_various/data_AvocadoPuree.csv"
        )

else:
    raise ValueError("Unsupported material")