import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from edit_distance_analysis import compute_stats_for_dataset

def plot_mean_comparison(all_model_data, output_dir, metric="tree"):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    
    models = list(all_model_data.keys())
    anchored_means = []
    direct_means = []
    anchored_errors = []
    direct_errors = []
    
    for model in models:
        data = all_model_data[model]
        anchored_mean = data["anchored_mean"]
        direct_mean = data["direct_mean"]
        anchored_problem_means = data["anchored_problem_means"]
        direct_problem_means = data["direct_problem_means"]
        
        anchored_means.append(anchored_mean)
        direct_means.append(direct_mean)
        
        if anchored_mean is not None and len(anchored_problem_means) > 0:
            anchored_std = np.std(anchored_problem_means, ddof=1)
            anchored_n = len(anchored_problem_means)
            anchored_error = anchored_std / np.sqrt(anchored_n)
        else:
            anchored_error = 0
        
        if direct_mean is not None and len(direct_problem_means) > 0:
            direct_std = np.std(direct_problem_means, ddof=1)
            direct_n = len(direct_problem_means)
            direct_error = direct_std / np.sqrt(direct_n)
        else:
            direct_error = 0
        
        anchored_errors.append(anchored_error)
        direct_errors.append(direct_error)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    direct_color = "#519ABA"
    anchored_color = "#FF5A78"
    
    ax.bar(x - width/2, direct_means, width, label="Direct", color=direct_color,
           yerr=direct_errors, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5}, alpha=0.85)
    ax.bar(x + width/2, anchored_means, width, label="Contextual Drag (2F)", color=anchored_color, 
           yerr=anchored_errors, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5}, alpha=0.85)
    
    all_values = []
    for i in range(len(models)):
        if anchored_means[i] is not None:
            all_values.append(anchored_means[i] + anchored_errors[i])
            all_values.append(anchored_means[i] - anchored_errors[i])
        if direct_means[i] is not None:
            all_values.append(direct_means[i] + direct_errors[i])
            all_values.append(direct_means[i] - direct_errors[i])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
        text_offset = 0.02 * y_range
    else:
        text_offset = 0
    
    def shorten_model_name(name):
        name = name.replace("_", "-")
        if name.startswith("GPT-OSS"):
            return name.replace("GPT-OSS", "G")
        elif name.startswith("Nemotron"):
            return name.replace("Nemotron", "N")
        elif name.startswith("Qwen3"):
            return name.replace("Qwen3", "Q")
        return name
    
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel(f"Mean Tree Edit Distance\nw.r.t. Draft in Context", fontsize=14)
    ax.set_xticks(x)
    model_labels = [shorten_model_name(m) for m in models]
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.tick_params(axis="y", which="major", labelsize=12)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)
    
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="gray", axis="y")
    
    for i, (anchored, direct) in enumerate(zip(anchored_means, direct_means)):
        if anchored is not None:
            ax.text(i + width/2, anchored + anchored_errors[i] + text_offset, f"{anchored:.2f}", 
                   ha="center", va="bottom", fontsize=10)
        if direct is not None:
            ax.text(i - width/2, direct + direct_errors[i] + text_offset, f"{direct:.2f}", 
                   ha="center", va="bottom", fontsize=10)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        fontsize=13,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        facecolor="white",
    )
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"edit_distance_comparison_all_models2.pdf"), bbox_inches="tight")
    plt.close()

def main():
    cache_dir = "cache/direct"
    output_dir = "figures"
    
    json_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.json')])
    all_model_means = {}
    
    for json_file in json_files:
        model_name = json_file.replace('.json', '')
        file_path = os.path.join(cache_dir, json_file)
        
        print(f"Processing {model_name}...")
        
        with open(file_path, 'r') as f:
            processed_data = json.load(f)
        
        stats = compute_stats_for_dataset(processed_data, metric="tree") #, filter_verification=True)
        print(f"  Stats computed: {len(stats)} entries")
        
        anchored_problem_means = [s["anchored_responses"] for s in stats if s["anchored_responses"] is not None]
        direct_problem_means = [s["init_response"] for s in stats if s["init_response"] is not None]
        
        anchored_mean = np.mean(anchored_problem_means) if len(anchored_problem_means) > 0 else None
        direct_mean = np.mean(direct_problem_means) if len(direct_problem_means) > 0 else None
        
        print(f"  Anchored mean: {anchored_mean:.2f}" if anchored_mean is not None else "  Anchored mean: None")
        print(f"  Direct mean: {direct_mean:.2f}" if direct_mean is not None else "  Direct mean: None")
        print(f"  Number of problems (anchored): {len(anchored_problem_means)}")
        print(f"  Number of problems (direct): {len(direct_problem_means)}")
        
        all_model_means[model_name] = {
            "anchored_mean": anchored_mean,
            "direct_mean": direct_mean,
            "anchored_problem_means": anchored_problem_means,
            "direct_problem_means": direct_problem_means
        }
    
    plot_mean_comparison(all_model_means, output_dir, metric="tree")

if __name__ == "__main__":
    main()
