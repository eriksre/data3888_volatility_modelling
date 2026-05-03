import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "Optiver_additional data/order_book_feature.csv",
    sep="\t",
)

df = df[df["stock_id"] == 22729].copy()

df["wap"] = (
    df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
) / (df["bid_size1"] + df["ask_size1"])

df = df.sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)

fig, axes = plt.subplots(figsize=(14, 5))

axes.plot(df.index, df["wap"], linewidth=0.8, color="steelblue")

breaks = df.groupby("time_id").apply(lambda g: g.index[0]).values
for x in breaks[1:]:
    axes.axvline(x=x, color="red", linewidth=0.5, alpha=0.5, linestyle="--")

axes.set_title("Weighted Average Price — Stock 22729 (chronological)", fontsize=13)
axes.set_xlabel("Observation index (sorted by time_id → seconds_in_bucket)")
axes.set_ylabel("WAP")
axes.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig("wap_stock_8382.png", dpi=150)
plt.show()

