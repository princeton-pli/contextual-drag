import os
import matplotlib.pyplot as plt
import numpy as np


def create_verification_visualizations(statistics, summary_stats, ds, output_path):
    """
    Create comprehensive visualizations for verification statistics.

    Args:
        statistics: Statistics dictionary by source and label
        summary_stats: Summary statistics dictionary
        ds: Dataset object
        output_path: Output directory path
    """
    # Extract summary stats
    total_predictions = summary_stats['total_predictions']
    valid_predictions = summary_stats['valid_predictions']
    invalid_predictions = summary_stats['invalid_predictions']
    tp_count = summary_stats['tp_count']
    tn_count = summary_stats['tn_count']
    fp_count = summary_stats['fp_count']
    fn_count = summary_stats['fn_count']
    sorted_sources = summary_stats['sorted_sources']
    sorted_labels = summary_stats['sorted_labels']
    fp_fn_entries = summary_stats['fp_fn_entries']

    # Prepare data for plotting
    sources = sorted_sources
    labels = sorted_labels

    # Extract fp and fn rates for plotting
    fp_rates = []
    fn_rates = []

    for source in sources:
        source_fp = []
        source_fn = []
        for label in labels:
            if label in statistics[source]:
                source_fp.append(statistics[source][label]['fp_rate'])
                source_fn.append(statistics[source][label]['fn_rate'])
            else:
                source_fp.append(0)  # Default to 0 if no data
                source_fn.append(0)
        fp_rates.append(source_fp)
        fn_rates.append(source_fn)

    # Set up the plot with 6 subplots (3x2) with increased size and margins
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(22, 16))

    # Add more space between subplots and around the figure
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.90, wspace=0.25, hspace=0.35)

    # Add main title
    fig.suptitle('Math Verification Performance Analysis', fontsize=14, fontweight='bold', y=0.95)

    # Bar width and positions
    bar_width = 0.15
    x = np.arange(len(sources))

    # Plot False Positive rates
    for i, label in enumerate(labels):
        label_fp_rates = [fp_rates[j][i] for j in range(len(sources))]
        ax1.bar(x + i * bar_width, label_fp_rates, bar_width, label=f'Label {label}')

    ax1.set_ylabel('FP Rate', fontsize=10)
    ax1.set_xlabel('Source', fontsize=9)
    ax1.set_title('FP Rates by Source\n(FP/(FP+TN))', fontsize=11, pad=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=9)
    ax1.set_xticks(x + bar_width * (len(labels) - 1) / 2)
    ax1.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)

    # Plot False Negative rates
    for i, label in enumerate(labels):
        label_fn_rates = [fn_rates[j][i] for j in range(len(sources))]
        ax2.bar(x + i * bar_width, label_fn_rates, bar_width, label=f'Label {label}')

    ax2.set_ylabel('FN Rate', fontsize=10)
    ax2.set_xlabel('Source', fontsize=9)
    ax2.set_title('FN Rates by Source\n(FN/(FN+TP))', fontsize=11, pad=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=9)
    ax2.set_xticks(x + bar_width * (len(labels) - 1) / 2)
    ax2.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)

    # Plot overall frequency statistics as a bar chart
    if valid_predictions > 0:
        categories = ['True Positives', 'True Negatives',
                     'False Positives', 'False Negatives']
        counts = [tp_count, tn_count, fp_count, fn_count]
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFA500']  # Green, Blue, Red, Orange

        bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_xlabel('Classification Type', fontsize=9)
        ax3.set_title('Classification Counts', fontsize=11, pad=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=9)
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels([cat.replace('\n', ' ') for cat in categories], rotation=45, ha='right', fontsize=8)

        # Add value labels on bars with better positioning
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.02,
                    f'{count:,}\n({count/valid_predictions*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Plot confusion matrix heatmap
        # Standard format: Rows=Actual, Columns=Predicted
        confusion_matrix = np.array([[tp_count, fn_count],  # Row 0: Actually Correct
                                   [fp_count, tn_count]])  # Row 1: Actually Incorrect

        im = ax4.imshow(confusion_matrix, interpolation='nearest', cmap='Blues', alpha=0.8)
        ax4.set_title('Confusion Matrix', fontsize=11, pad=10)
        ax4.set_xlabel('Predicted', fontsize=9)
        ax4.set_ylabel('Actual', fontsize=9)
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Correct', 'Incorrect'], fontsize=9)
        ax4.set_yticklabels(['Correct', 'Incorrect'], fontsize=9)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, f'{confusion_matrix[i, j]:,}\n({confusion_matrix[i, j]/valid_predictions*100:.1f}%)',
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')

        # Add colorbar to the same figure as the confusion matrix
        cbar = fig.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Count')

    # Plot stepwise verification verdict distribution (bottom-left: ax5)
    if total_predictions > 0:
        # Collect verdict frequencies
        verdict_counts = {}
        for entry in ds:
            if 'stepwise_verification_verdict' in entry:
                verdict = entry['stepwise_verification_verdict']
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        if verdict_counts:
            verdicts = list(verdict_counts.keys())
            counts = list(verdict_counts.values())

            # Sort by frequency for better visualization
            verdict_count_pairs = sorted(zip(counts, verdicts), reverse=True)
            counts, verdicts = zip(*verdict_count_pairs)

            # Create color mapping for different verdict types
            color_map = {
                'correct': '#2E8B57',      # Green
                'incorrect': '#FF6347',    # Red
                'partially_correct': '#FFA500',  # Orange
                'unparsable_verdict': '#DC143C', # Crimson
                'parsable_ood_verdict': '#4169E1' # Blue
            }

            colors = [color_map.get(verdict, '#808080') for verdict in verdicts]  # Gray for unknown

            bars = ax5.bar(verdicts, counts, color=colors, alpha=0.7)
            ax5.set_ylabel('Count', fontsize=10)
            ax5.set_xlabel('Verdict Type', fontsize=10)
            ax5.set_title('Verdict Distribution', fontsize=11, pad=10)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='both', labelsize=9)

            # Add value labels on bars with better positioning
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.03,
                        f'{count:,}\n({count/total_predictions*100:.1f}%)',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

            # Rotate x-axis labels for better readability
            ax5.set_xticks(range(len(verdicts)))
            ax5.set_xticklabels(verdicts, rotation=45, ha='right', fontsize=8)

    # Create detailed breakdown bar chart (bottom-right: ax6)
    if total_predictions > 0:
        missing_prediction_only = summary_stats.get('missing_prediction_only', 0)
        missing_ground_truth_only = summary_stats.get('missing_ground_truth_only', 0)
        both_missing = summary_stats.get('both_missing', 0)

        categories = ['Valid\nPredictions', 'Missing\nPrediction\nOnly', 'Missing\nGround Truth\nOnly', 'Both\nMissing']
        counts = [valid_predictions, missing_prediction_only, missing_ground_truth_only, both_missing]
        colors_detail = ['#2E8B57', '#FF6347', '#FFA500', '#DC143C']

        bars = ax6.bar(categories, counts, color=colors_detail, alpha=0.7)
        ax6.set_ylabel('Count', fontsize=10)
        ax6.set_xlabel('Validity Type', fontsize=10)
        ax6.set_title('Validity Breakdown', fontsize=11, pad=10)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='both', labelsize=9)

        # Add value labels on bars with better positioning
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.03,
                    f'{count:,}\n({count/total_predictions*100:.1f}%)',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Rotate x-axis labels for better readability
        ax6.set_xticks(range(len(categories)))
        ax6.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)

    # Save the figure (layout already adjusted with subplots_adjust)
    plt.savefig(os.path.join(output_path, 'fp_fn_rates_by_source_label.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the main figure

    # Note: Confusion matrix is included in the main visualization
    # No separate confusion matrix file needed since it's part of the 4-panel plot
