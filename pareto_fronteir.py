import matplotlib.pyplot as plt
import numpy as np

models = {
    'HAR-RV':          {'time': 0.043,    'qlike': 0.883},
    'LASSO':           {'time': 0.240,    'qlike': 0.900},
    'Ridge':           {'time': 0.309,    'qlike': 0.922},
    'Decision Tree':   {'time': 0.322,    'qlike': 0.691},
    'XGBoost':         {'time': 0.713,    'qlike': 0.661},
    'Random Forest':   {'time': 5.478,    'qlike': 0.696},
    'Linear Reg.':     {'time': 8.576,    'qlike': 0.925},
    'GARCH(1,1)':      {'time': 4666.091, 'qlike': 0.632},
    'GJR-GARCH(1,1)':  {'time': 5912.321, 'qlike': 0.596},
}


def compute_pareto_frontier(models: dict) -> list[str]:
    """
    Returns names of Pareto-optimal models.
    A model is Pareto optimal if no other model has both
    lower (better) QLIKE and lower (better/faster) inference time.
    """
    pareto = []
    names = list(models.keys())
    for name, m in models.items():
        dominated = False
        for other_name, other in models.items():
            if other_name == name:
                continue
            if other['time'] <= m['time'] and other['qlike'] <= m['qlike']:
                if other['time'] < m['time'] or other['qlike'] < m['qlike']:
                    dominated = True
                    break
        if not dominated:
            pareto.append(name)
    return pareto


pareto_names = compute_pareto_frontier(models)
pareto_models = {k: models[k] for k in pareto_names}
dominated_models = {k: models[k] for k in models if k not in pareto_names}

# Sort frontier by inference time for line plotting
frontier_sorted = sorted(pareto_models.items(), key=lambda x: x[1]['time'])

fig, ax = plt.subplots(figsize=(10, 6))

# Frontier line
fx = [m['time'] for _, m in frontier_sorted]
fy = [m['qlike'] for _, m in frontier_sorted]
ax.plot(fx, fy, color='red', linewidth=1.5, label='Pareto frontier', zorder=2)

# Pareto optimal points
px = [m['time'] for m in pareto_models.values()]
py = [m['qlike'] for m in pareto_models.values()]
ax.scatter(px, py, color='red', s=60, zorder=4, label='Pareto optimal models')

# Dominated points
dx = [m['time'] for m in dominated_models.values()]
dy = [m['qlike'] for m in dominated_models.values()]
ax.scatter(dx, dy, color='gray', s=50, zorder=3, alpha=0.7, label='Dominated models')

# Labels
label_offsets = {
    'HAR-RV':          (-0.3,  0.008),
    'Decision Tree':   (0.03,  0.008),
    'XGBoost':         (0.03, -0.012),
    'GARCH(1,1)':      (-1500, 0.008),
    'GJR-GARCH(1,1)':  (100,  -0.012),
    'LASSO':           (0.03,  0.015),
    'Ridge':           (0.03, -0.020),
    'Random Forest':   (0.15, -0.012),
    'Linear Reg.':     (0.3,   0.008),
}

for name, m in models.items():
    ox, oy = label_offsets.get(name, (0.03, 0.008))
    color = 'red' if name in pareto_names else 'gray'
    ax.annotate(name, xy=(m['time'], m['qlike']),
                xytext=(m['time'] + ox, m['qlike'] + oy),
                fontsize=8, color=color)

# Latency threshold lines
ax.axvline(x=1,    color='green',  linestyle='--', linewidth=1.2, label='1 µs (HFT)')
ax.axvline(x=10,   color='gold',   linestyle='--', linewidth=1.2, label='10 µs (desk trading)')
ax.axvline(x=1000, color='purple', linestyle='--', linewidth=1.2, label='1000 µs')

ax.set_xscale('log')
ax.set_xlabel('Inference Time (µs)')
ax.set_ylabel('QLIKE (lower is better)')
ax.set_title('Pareto Frontier: Volatility Model Selection')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, which='both', linestyle=':', alpha=0.4)
ax.set_xlim(0.02, 20000)
ax.set_ylim(0.55, 0.96)

plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=150)
plt.show()