"""
Plot order book snapshot for a given stock + time bucket.

Shows:
  - Best bid (bid_price1)    — blue dashed
  - Best offer (ask_price1)  — orange dashed
  - WAP (1-level)            — steelblue solid
  - Buys                     — green upward tick  (^)
  - Sells                    — red downward tick  (v)

Trade direction is inferred via the tick rule:
  up-tick → buy, down-tick → sell, zero-tick → carry previous direction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ──────────────────────────────────────────────────────────────────
STOCK_ID = 8382
TIME_ID  = 13
# ────────────────────────────────────────────────────────────────────────────

# ── Load & filter order book ─────────────────────────────────────────────────
ob = pd.read_csv("Optiver_additional data/order_book_feature.csv", sep="\t")
ob = ob[(ob["stock_id"] == STOCK_ID) & (ob["time_id"] == TIME_ID)].copy()
ob = ob.sort_values("seconds_in_bucket").reset_index(drop=True)

if ob.empty:
    raise ValueError(f"No order book data for stock_id={STOCK_ID}, time_id={TIME_ID}")

# 1-level WAP: always stays within the best bid/ask spread
ob["wap1"] = (
    ob["bid_price1"] * ob["ask_size1"] + ob["ask_price1"] * ob["bid_size1"]
) / (ob["bid_size1"] + ob["ask_size1"])

# ── Load & filter trades ──────────────────────────────────────────────────────
tr = pd.read_csv("Optiver_additional data/trades.csv", sep="\t")
tr = tr[(tr["stock_id"] == STOCK_ID) & (tr["time_id"] == TIME_ID)].copy()
tr = tr.sort_values("seconds_in_bucket").reset_index(drop=True)

# Infer direction via tick rule
if not tr.empty:
    direction = []
    last = "buy"  # default for first trade
    prev_price = None
    for p in tr["price"]:
        if prev_price is None or p > prev_price:
            last = "buy"
        elif p < prev_price:
            last = "sell"
        # zero-tick → carry last direction
        direction.append(last)
        prev_price = p
    tr["direction"] = direction

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

t = ob["seconds_in_bucket"]

ax.step(t, ob["bid_price1"], color="royalblue",   lw=1.2, ls="--", where="post", label="Best bid")
ax.step(t, ob["ask_price1"], color="darkorange",  lw=1.2, ls="--", where="post", label="Best offer")
ax.step(t, ob["wap1"],       color="steelblue",    lw=1.5,         where="post", label="WAP")

# Shade bid-ask spread
ax.fill_between(t, ob["bid_price1"], ob["ask_price1"],
                alpha=0.08, color="grey", step="post", label="Spread")

# Overlay trades
if not tr.empty:
    buys  = tr[tr["direction"] == "buy"]
    sells = tr[tr["direction"] == "sell"]

    ax.scatter(buys["seconds_in_bucket"],  buys["price"],
               marker="^", color="limegreen", s=20,
               zorder=5, label="Buy")
    ax.scatter(sells["seconds_in_bucket"], sells["price"],
               marker="v", color="crimson",   s=20,
               zorder=5, label="Sell")

ax.set_title(
    f"Order Book & Trades — Stock {STOCK_ID}, Time bucket {TIME_ID}",
    fontsize=13,
)
ax.set_xlabel("Seconds in bucket")
ax.set_ylabel("Price")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

plt.tight_layout()
plt.show()
