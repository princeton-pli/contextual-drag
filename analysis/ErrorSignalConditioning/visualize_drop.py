import json
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Set font style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

TASK_DISPLAY = {
    'aime24': 'AIME24',
    'aime25': 'AIME25',
    'hmmt24': 'HMMT24',
    'hmmt25': 'HMMT25',
    'gpqa': 'GPQA',
    'mmlu': 'MMLU',
    'crux-i': 'CruxEval-I',
    '24-game': 'Game of 24',
}

# Create custom colormap from pink to white to blue (inverted)
colors = ['#FF5A78', 'white', '#519ABA']
cmap_middle = LinearSegmentedColormap.from_list('custom_gradient', colors)

# Load the data
with open('correct_verification_conditioning_results_prompted.json', 'r') as f:
    prompted_data = json.load(f)

with open('correct_verification_conditioning_results_self_detected.json', 'r') as f:
    self_detected_data = json.load(f)

# Extract benchmark and model names
benchmarks = list(prompted_data.keys())

# Define model order and display names
model_order = ['GPT_OSS_120B', 'GPT_OSS_20B', 'Nemotron_32B', 'Nemotron_7B', 'Qwen3_32B', 'Qwen3_8B']
model_display_names = {
    'GPT_OSS_120B': 'G-120B',
    'GPT_OSS_20B': 'G-20B',
    'Nemotron_32B': 'N-32B',
    'Nemotron_7B': 'N-7B',
    'Qwen3_32B': 'Q-32B',
    'Qwen3_8B': 'Q-8B'
}

models = model_order
model_labels = [model_display_names[model] for model in models]

# Initialize matrices for the differences
external_awareness = np.zeros((len(benchmarks), len(models)))
self_detected_awareness = np.zeros((len(benchmarks), len(models)))

# Compute differences for external awareness (prompted data)
for i, benchmark in enumerate(benchmarks):
    for j, model in enumerate(models):
        raw = prompted_data[benchmark][model]['correctness_raw']
        raw_init = prompted_data[benchmark][model]['correctness_raw_init_sampling']
        external_awareness[i, j] = raw - raw_init

# Compute differences for self_detected awareness (self_detected data)
for i, benchmark in enumerate(benchmarks):
    for j, model in enumerate(models):
        filtered = self_detected_data[benchmark][model]['correctness_filtered']
        filtered_init = self_detected_data[benchmark][model]['correctness_filtered_init_sampling']
        self_detected_awareness[i, j] = filtered - filtered_init

# Find the actual min/max values to avoid clipping, but keep zero-centered
all_values = np.concatenate([external_awareness.flatten(), self_detected_awareness.flatten()])
max_abs = np.max(np.abs(all_values))
# Add small padding to avoid clipping at extremes
max_abs = max_abs + 0.01
# Set symmetric bounds around zero
vmin = -max_abs
vmax = max_abs

# Create benchmark display labels
benchmark_labels = [TASK_DISPLAY.get(benchmark, benchmark) for benchmark in benchmarks]

# Create the figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")
ax2.set_facecolor("white")

# Plot external awareness (top)
im1 = ax1.imshow(external_awareness, cmap=cmap_middle, aspect='auto', vmin=vmin, vmax=vmax)
ax1.set_title('Prompted Error Awareness', fontsize=14)
# ax1.set_ylabel('Benchmark', fontsize=14, labelpad=-10)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(model_labels, fontsize=12)
ax1.set_yticks(range(len(benchmarks)))
ax1.set_yticklabels(benchmark_labels, fontsize=12)
ax1.tick_params(axis="x", which="major", labelsize=12)
ax1.tick_params(axis="y", which="major", labelsize=12)

# Style spines for ax1
for spine in ax1.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

# Add values to the cells
for i in range(len(benchmarks)):
    for j in range(len(models)):
        val = external_awareness[i, j] * 100
        val_str = f'+{val:.1f}%' if val > 0 else f'{val:.1f}%'
        text = ax1.text(j, i, val_str,
                       ha='center', va='center', color='black', fontsize=10)

# Plot self_detected awareness (bottom)
im2 = ax2.imshow(self_detected_awareness, cmap=cmap_middle, aspect='auto', vmin=vmin, vmax=vmax)
ax2.set_title('Self-detected Error Awareness', fontsize=14)
ax2.set_xlabel('Model', fontsize=14)
# ax2.set_ylabel('Benchmark', fontsize=14, labelpad=-10)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(model_labels, fontsize=12)
ax2.set_yticks(range(len(benchmarks)))
ax2.set_yticklabels(benchmark_labels, fontsize=12)
ax2.tick_params(axis="x", which="major", labelsize=12)
ax2.tick_params(axis="y", which="major", labelsize=12)

# Style spines for ax2
for spine in ax2.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

# Add values to the cells
for i in range(len(benchmarks)):
    for j in range(len(models)):
        val = self_detected_awareness[i, j] * 100
        val_str = f'+{val:.1f}%' if val > 0 else f'{val:.1f}%'
        text = ax2.text(j, i, val_str,
                       ha='center', va='center', color='black', fontsize=10)

# Adjust subplot parameters to make room for colorbar at top
plt.subplots_adjust(top=0.88, hspace=0.3)

# Add colorbar on the top (moved upward)
cbar_ax = fig.add_axes([0.15, 0.965, 0.7, 0.02])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
# Add label above the colorbar manually
fig.text(0.5, 0.995, 'Absolute Performance Change (%)', fontsize=14, ha='center', va='bottom')
# Update colorbar ticks to show percentages - symmetric around zero
num_ticks = 7
tick_values = np.linspace(-max_abs, max_abs, num_ticks)
tick_values = np.round(tick_values, 2)
cbar.set_ticks(tick_values)
cbar.set_ticklabels([f'{t*100:+.0f}%' if t != 0 else '0%' for t in tick_values])

# Style colorbar spines
for spine in cbar_ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

plt.savefig('awareness_visualization.pdf', dpi=300, bbox_inches='tight')
print('Saved figure to awareness_visualization.pdf')
plt.close()