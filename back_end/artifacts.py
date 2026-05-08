import matplotlib.pyplot as plt


def plot_pacf(pacf_df, conf_int, stock_name):

    plt.figure(figsize=(10, 5))

    plt.vlines(
        pacf_df["lag"],
        ymin=0,
        ymax=pacf_df["value"]
    )

    plt.axhline(0)

    plt.axhline(
        conf_int,
        linestyle="dashed"
    )

    plt.axhline(
        -conf_int,
        linestyle="dashed"
    )

    plt.title(
        f"Average PACF of Squared Log Returns ({stock_name})"
    )

    plt.xlabel("Lag")
    plt.ylabel("PACF")

    plt.show()


def plot_acf(acf_df, conf_int, stock_name):

    plt.figure(figsize=(10, 5))

    plt.vlines(
        acf_df["lag"],
        ymin=0,
        ymax=acf_df["value"]
    )

    plt.axhline(0)

    plt.axhline(
        conf_int,
        linestyle="dashed"
    )

    plt.axhline(
        -conf_int,
        linestyle="dashed"
    )

    plt.title(
        f"Average ACF of Squared Log Returns ({stock_name})"
    )

    plt.xlabel("Lag")
    plt.ylabel("ACF")

    plt.show()