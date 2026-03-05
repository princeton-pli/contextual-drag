import json
import os
import copy
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def compute_fp_fn_rates(pred, ans):
    """
    Compute false positive and false negative rates.

    Args:
        pred: Boolean array of predictions
        ans: Boolean array of ground truth answers

    Returns:
        tuple: (fp_rate, fn_rate)
    """
    # pred and ans are both boolean arrays
    # return the fp and fn rates
    tp = np.sum((pred == ans) & (ans == True))
    fp = np.sum((pred == ans) & (ans == False))
    tn = np.sum((pred != ans) & (ans == False))
    fn = np.sum((pred != ans) & (ans == True))

    # Debug: print counts for first few calls to check logic
    if not hasattr(compute_fp_fn_rates, 'debug_count'):
        compute_fp_fn_rates.debug_count = 0

    if compute_fp_fn_rates.debug_count < 3:  # Only print first 3 calls
        total = len(pred) if hasattr(pred, '__len__') else 0
        print(f"DEBUG compute_fp_fn_rates call {compute_fp_fn_rates.debug_count + 1}:")
        print(f"  Total samples: {total}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  FP rate: {fp / (fp + tn) if (fp + tn) > 0 else 0:.3f}")
        print(f"  FN rate: {fn / (fn + tp) if (fn + tp) > 0 else 0:.3f}")
        print(f"  pred True/False counts: {np.sum(pred == True)}/{np.sum(pred == False)}")
        print(f"  ans True/False counts: {np.sum(ans == True)}/{np.sum(ans == False)}")
        print()
        compute_fp_fn_rates.debug_count += 1

    return fp / (fp + tn), fn / (fn + tp)


def sample_fp_fn_entries(fp_fn_entries, ds, output_path, sample_size=50):
    """
    Sample FP and FN entries evenly and save to a cleaned JSON file with separate sections.

    Args:
        fp_fn_entries: Dict with 'fp' and 'fn' keys containing lists of entries
        ds: The dataset object to access full entries
        output_path: Directory to save the JSON file
        sample_size: Total number of entries to sample (will be split between FP and FN)

    Output format:
        - metadata: Statistics about available and sampled entries
        - false_positives: Array of false positive samples
        - false_negatives: Array of false negative samples

        Each sample contains:
        - question: The math problem
        - answer: The expected answer
        - model_initial_response: The model's initial solution
        - model_stepwise_verification_response: The verification analysis
        - answer_correctness: Boolean indicating if initial answer was correct
        - verdict: Final verification verdict
        - source: Dataset source
        - label: Difficulty/skill label
        - predicted: What the verifier predicted
        - ground_truth: The actual correctness
    """
    import random

    fp_entries = fp_fn_entries['fp']
    fn_entries = fp_fn_entries['fn']

    print(f"Found {len(fp_entries)} false positive entries and {len(fn_entries)} false negative entries")

    # Calculate sample sizes for each category
    total_available = len(fp_entries) + len(fn_entries)
    if total_available == 0:
        print("No FP or FN entries found, skipping sampling")
        return

    # Sample evenly between FP and FN
    fp_sample_size = min(len(fp_entries), sample_size // 2)
    fn_sample_size = min(len(fn_entries), sample_size - fp_sample_size)

    # If one category has fewer entries, give the remaining budget to the other
    if fp_sample_size < sample_size // 2 and fn_sample_size < len(fn_entries):
        fn_sample_size = min(len(fn_entries), sample_size - fp_sample_size)
    elif fn_sample_size < sample_size // 2 and fp_sample_size < len(fp_entries):
        fp_sample_size = min(len(fp_entries), sample_size - fn_sample_size)

    # Sample entries
    sampled_fp = random.sample(fp_entries, fp_sample_size) if fp_sample_size > 0 else []
    sampled_fn = random.sample(fn_entries, fn_sample_size) if fn_sample_size > 0 else []

    # Add cleaned dataset entries to sampled data
    def clean_entry(entry):
        entry_index = entry['index']
        full_entry = dict(ds[entry_index])

        # Extract only the required fields
        cleaned = {
            'question': full_entry.get('problem', ''),
            'answer': full_entry.get('answer', ''),
            'model_initial_response': full_entry.get('init_response_generations_generated_response', ''),
            'model_stepwise_verification_response': full_entry.get('stepwise_verification_generations_generated_response', ''),
            'answer_correctness': full_entry.get('init_response_generations_correctness', None),
            'verdict': full_entry.get('stepwise_verification_verdict', ''),
            'source': entry.get('source', ''),
            'label': entry.get('label', ''),
            'predicted': entry.get('predicted', None),
            'ground_truth': entry.get('ground_truth', None)
        }
        return cleaned

    # Clean FP entries
    cleaned_fp = [clean_entry(entry) for entry in sampled_fp]

    # Clean FN entries
    cleaned_fn = [clean_entry(entry) for entry in sampled_fn]

    # Prepare output data with separate FP and FN sections
    output_data = {
        'metadata': {
            'total_fp_available': len(fp_entries),
            'total_fn_available': len(fn_entries),
            'fp_sampled': len(cleaned_fp),
            'fn_sampled': len(cleaned_fn),
            'total_sampled': len(cleaned_fp) + len(cleaned_fn),
            'sample_timestamp': str(os.path.basename(output_path))
        },
        'false_positives': cleaned_fp,
        'false_negatives': cleaned_fn
    }

    # Save to JSON file
    output_file = os.path.join(output_path, 'fp_fn_samples.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(cleaned_fp) + len(cleaned_fn)} sampled entries to {output_file}")
    print(f"  - False Positives: {len(cleaned_fp)} (from {len(fp_entries)} available)")
    print(f"  - False Negatives: {len(cleaned_fn)} (from {len(fn_entries)} available)")


def compute_statistics_for_groups(ds, output_path, sample_size=50):
    """
    Compute statistics for different source-label combinations.

    Args:
        ds: Dataset object
        output_path: Output directory path
        sample_size: Number of FP/FN samples to save

    Returns:
        dict: Statistics dictionary
    """
    source_array = ds['source']
    all_sources = set(source_array)
    label_array = ds['label']
    all_labels = set(label_array)

    sorted_sources = copy.deepcopy(list(all_sources))
    sorted_labels = copy.deepcopy(list(all_labels))
    sorted_sources.sort()
    sorted_labels.sort()

    print(f"Processing {len(all_sources)} sources and {len(all_labels)} labels")

    # Get all required columns once
    boolean_verdicts = ds['stepwise_verification_boolean_verdict']
    ground_truth_correctness = ds['init_response_generations_correctness']

    # Group data by source-label combinations in a single pass
    grouped_data = {}
    fp_fn_entries = {'fp': [], 'fn': []}

    for i in tqdm(range(0, len(ds), max(1, len(ds) // 20000))):
        source = source_array[i]
        label = label_array[i]
        key = (source, label)
        pred = boolean_verdicts[i]
        true = ground_truth_correctness[i]

        if pred is None:
            pred = False
        if true is None:
            true = False

        if key not in grouped_data:
            grouped_data[key] = {
                'boolean_verdicts': [],
                'ground_truth_correctness': [],
                'indices': []
            }

        grouped_data[key]['boolean_verdicts'].append(pred)
        grouped_data[key]['ground_truth_correctness'].append(true)
        grouped_data[key]['indices'].append(i)

        # Collect FP and FN entries
        if pred is not None and true is not None:
            if pred == True and true == False:  # False Positive
                if len(fp_fn_entries['fp']) < sample_size:
                    fp_fn_entries['fp'].append({
                        'index': i,
                        'source': source,
                        'label': label,
                        'predicted': pred,
                        'ground_truth': true
                    })
            elif pred == False and true == True:  # False Negative
                if len(fp_fn_entries['fn']) < sample_size:
                    fp_fn_entries['fn'].append({
                        'index': i,
                        'source': source,
                        'label': label,
                        'predicted': pred,
                        'ground_truth': true
                    })

    def compute_stats_for_group(key, group_data):
        source, label = key
        print(f"Processing source: {source}, label: {label}")
        boolean_verdict = np.array(group_data['boolean_verdicts'])
        ground_truth = np.array(group_data['ground_truth_correctness'])
        fp_rate, fn_rate = compute_fp_fn_rates(boolean_verdict, ground_truth)
        return source, label, fp_rate, fn_rate

    # Process all groups in parallel
    stats = Parallel(n_jobs=-1)(
        delayed(compute_stats_for_group)(key, group_data)
        for key, group_data in grouped_data.items()
    )

    # Build statistics dictionary
    statistics = {}
    for source, label, fp_rate, fn_rate in stats:
        if source not in statistics:
            statistics[source] = {}
        statistics[source][label] = {
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
        }

    # Compute overall frequency statistics
    print("\n=== Overall Verification Statistics ===")
    total_predictions = len(boolean_verdicts)
    valid_predictions = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                           if p is not None and t is not None)

    # Calculate invalid prediction breakdown
    missing_prediction_only = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                                 if p is None and t is not None)
    missing_ground_truth_only = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                                   if p is not None and t is None)
    both_missing = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                      if p is None and t is None)
    invalid_predictions = missing_prediction_only + missing_ground_truth_only + both_missing

    # Calculate TP, TN, FP, FN counts
    tp_count = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                   if p is True and t is True)
    tn_count = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                   if p is False and t is False)
    fp_count = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                   if p is True and t is False)
    fn_count = sum(1 for p, t in zip(boolean_verdicts, ground_truth_correctness)
                   if p is False and t is True)

    # Calculate percentages
    if valid_predictions > 0:
        tp_percent = (tp_count / valid_predictions) * 100
        tn_percent = (tn_count / valid_predictions) * 100
        fp_percent = (fp_count / valid_predictions) * 100
        fn_percent = (fn_count / valid_predictions) * 100

        print(f"Total predictions: {total_predictions}")
        print(f"Valid predictions: {valid_predictions} ({valid_predictions/total_predictions*100:.1f}%)")
        print(f"Invalid predictions: {invalid_predictions} ({invalid_predictions/total_predictions*100:.1f}%)")
        print(f"  - Missing prediction only: {missing_prediction_only} ({missing_prediction_only/total_predictions*100:.1f}%)")
        print(f"  - Missing ground truth only: {missing_ground_truth_only} ({missing_ground_truth_only/total_predictions*100:.1f}%)")
        print(f"  - Both missing: {both_missing} ({both_missing/total_predictions*100:.1f}%)")
        print(f"\nClassification Results (based on {valid_predictions} valid predictions):")
        print(f"  True Positives (TP):  {tp_count:5d} ({tp_percent:5.1f}%) - Correct solutions classified as correct")
        print(f"  True Negatives (TN):  {tn_count:5d} ({tn_percent:5.1f}%) - Incorrect solutions classified as incorrect")
        print(f"  False Positives (FP): {fp_count:5d} ({fp_percent:5.1f}%) - Incorrect solutions classified as correct")
        print(f"  False Negatives (FN): {fn_count:5d} ({fn_percent:5.1f}%) - Correct solutions classified as incorrect")

        # Calculate rates (consistent with bar plots)
        fp_rate = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0
        fn_rate = fn_count / (fn_count + tp_count) if (fn_count + tp_count) > 0 else 0

        print(f"\nError Rates (consistent with bar plots):")
        print(f"  False Positive Rate: {fp_rate:.3f} (FP/(FP+TN)) - Proportion of predicted positives that were wrong")
        print(f"  False Negative Rate: {fn_rate:.3f} (FN/(FN+TP)) - Proportion of actual positives that were missed")

        # Calculate accuracy and other metrics
        accuracy = (tp_count + tn_count) / valid_predictions * 100
        precision = tp_count / (tp_count + fp_count) * 100 if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) * 100 if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.1f}%")
        print(f"  Precision: {precision:.1f}%")
        print(f"  Recall:    {recall:.1f}%")
        print(f"  F1-Score:  {f1_score:.1f}%")
    else:
        print("No valid predictions found!")

    # Sample and save FP/FN entries
    sample_fp_fn_entries(fp_fn_entries, ds, output_path, sample_size)

    return statistics, {
        'total_predictions': total_predictions,
        'valid_predictions': valid_predictions,
        'invalid_predictions': invalid_predictions,
        'tp_count': tp_count,
        'tn_count': tn_count,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'sorted_sources': sorted_sources,
        'sorted_labels': sorted_labels,
        'fp_fn_entries': fp_fn_entries
    }
