"""
Top-K Sensor Identification Metric for Explainable Anomaly Detection.

This metric evaluates whether the ground-truth attacked sensors
are correctly identified within the top-K explained features.
"""

from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import pandas as pd


# Ground-truth attacked sensors keyed by dataset and attack number.
ATTACKED_SENSORS_BY_DATASET = {
    "swat": {
        "1": ["MV101"],
        "2": ["P102"],
        "3": ["LIT101"],
        "4": ["MV504"],
        "6": ["AIT202"],
        "7": ["LIT301"],
        "8": ["DPIT301"],
        "10": ["FIT401"],
        "11": ["FIT401"],
        "13": ["MV304"],
        "14": ["MV303"],
        "16": ["LIT301"],
        "17": ["MV303"],
        "19": ["AIT504"],
        "20": ["AIT504"],
        "21": ["MV101", "LIT101"],
        "22": ["UV401", "AIT502", "P501"],
        "23": ["P602", "DIT301", "MV302"],
        "24": ["P203", "P205"],
        "25": ["LIT401", "P401"],
        "26": ["P101", "LIT301"],
        "27": ["P302", "LIT401"],
        "28": ["P302"],  # Attack 28 is long.
        "29": ["P201", "P203", "P205"],
        "30": ["LIT101", "P101", "MV201"],
        "31": ["LIT401"],
        "32": ["LIT301"],
        "33": ["LIT101"],
        "34": ["P101"],
        "35": ["P101", "P102"],
        "36": ["LIT101"],
        "37": ["P501", "FIT502"],
        "38": ["AIT402", "AIT502"],
        "39": ["FIT401", "AIT502"],
        "40": ["FIT401"],
        "41": ["LIT301"],
    },
    "wadi": {
        "1": ["1_MV_001_STATUS"],
        "2": ["1_FIT_001_PV"],
        "3": ["2_LIT_002"],
        "4": ["1_AIT_001_PV"],
        "5": [
            "2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO",
            "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO",
        ],
        "6": ["2_MCV_101_CO", "2_MCV_201_CO"],
        "7": ["1_AIT_002_PV", "2_MV_003_STATUS"],
        "8": ["2_MCV_007_CO"],
        "9": ["1_P_006_STATUS"],
        "10": ["1_MV_001_STATUS"],
        "11": ["2_MCV_007_CO"],
        "12": ["2_MCV_007_CO"],
        "13": ["2_PIC_003_SP", "2_PIC_003_PV"],
        "14": ["1_P_001_STATUS", "1_P_003_STATUS"],
        "15": ["2_LIT_002"],  # Ground-truth sensor is absent from the provided WADI data frame.
    },
}

ATTACKED_SENSORS_DICT = ATTACKED_SENSORS_BY_DATASET["swat"]


def get_attack_sensors(dataset: str) -> dict:
    """Select and return the active ground-truth mapping for a dataset."""
    global ATTACKED_SENSORS_DICT
    try:
        selected = ATTACKED_SENSORS_BY_DATASET[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown attack-sensor dataset: {dataset}") from error
    ATTACKED_SENSORS_DICT = selected
    return selected


def get_top_k_sensors_with_scores(
    ranked_features: List[Tuple[str, float]],
    k: int
) -> List[Tuple[str, float]]:
    ranked_features_sorted = sorted(
        ranked_features,
        key=lambda x: x[1],
        reverse=True
    )
    return ranked_features_sorted[:k]


def compute_topk_score_for_key(
    ranked_features: list,
    key: str,
    k: int = 5,
    attack_sensors_dataset: str = "swat",
) -> dict:

    # Fetch ground-truth attacked sensors
    ground_truth = get_attack_sensors(attack_sensors_dataset).get(key, [])
    attacked_sensors = set(ground_truth)

    top_k = get_top_k_sensors_with_scores(ranked_features, k)

    top_k_names = [sensor for sensor, score in top_k]

    # Compute weighted Top-K hit
    hits = set(top_k_names) & attacked_sensors
    weighted_score = len(hits) / len(attacked_sensors) if attacked_sensors else 0.0

    # Confidence-weighted score
    matched_scores = [score for sensor, score in top_k if sensor in hits]
    weighted_score_with_confidence = weighted_score * sum(matched_scores) / 100

    return {
        "weighted_top_k_hit": weighted_score,
        "weighted_score_with_confidence": weighted_score_with_confidence,
        "top_k_sensors": top_k_names,
        "attacked_sensors": ground_truth
    }


def compute_topk_score_for_all_keys(
    all_ranked_features: Dict[str, List[tuple]],  # {key: ranked_features_list}
    k: int = 5,
    normalize: bool = True,
    attack_sensors_dataset: str = "swat",
) -> Dict[str, float]:
    """
    Compute aggregated weighted Top-K hit metrics for multiple attacks.

    Parameters
    ----------
    all_ranked_features : dict
        Dictionary mapping key (attack ID) to ranked feature list.
        Example: {"1": [("FIT101", 0.1), ("LIT101", 0.05), ...], ...}
    k : int
        Top-K cutoff for each key.
    normalize : bool
        If True, divide summed metric by number of keys to get average score.

    Returns
    -------
    dict
        Aggregated metrics:
        - "avg_weighted_top_k_hit"
        - "avg_weighted_score_with_confidence"
    """
    total_weighted_hit = 0.0
    total_weighted_conf = 0.0
    num_keys = len(all_ranked_features)
    per_key = {}

    for key, ranked_features in all_ranked_features.items():
        scores = compute_topk_score_for_key(
            ranked_features, key, k, attack_sensors_dataset=attack_sensors_dataset
        )

        per_key[key] = {
            f"Top_K{k}_Recall": scores["weighted_top_k_hit"],
            f"Top_K{k}_Recall_Confidence": scores["weighted_score_with_confidence"],
            "top_k_sensors": scores.get("top_k_sensors", []),
            "attacked_sensors": scores.get("attacked_sensors", []),
        }

        total_weighted_hit += scores["weighted_top_k_hit"]
        total_weighted_conf += scores["weighted_score_with_confidence"]

    if normalize and num_keys > 0:
        total_weighted_hit /= num_keys
        total_weighted_conf /= num_keys

    return {
        f"Top_K{k}_Recall": total_weighted_hit,
        f"Top_K{k}_Recall_Confidence": total_weighted_conf,
        f"Top_K{k}_Per_Key": per_key,
    }




###########################################################
############### TIME ANOMALY METRICS BELOW ################
###########################################################

def weighted_top_k_hit_from_ranked_features(
    ranked_features: List[Tuple[str, float]],
    attacked_sensors: set,
    k: int = 5
) -> float:
    # sort + top-k
    top_k = get_top_k_sensors_with_scores(ranked_features, k)
    top_k_names = [s for s, _ in top_k]
    hits = set(top_k_names) & attacked_sensors
    return (len(hits) / len(attacked_sensors)) if attacked_sensors else 0.0


def early_root_cause_metrics_for_key(
    ranked_features_per_window: List[List[Tuple[str, float]]],  # length N
    key: str,
    k: int = 5,
    mode: str = "any",  # "any" | "all" | "threshold"
    attack_sensors_dataset: str = "swat",
) -> Dict[str, float]:
    """
    Computes early-identification metrics for one anomaly sample (one key).

    mode:
      - "any": first window where weighted_hit > 0
      - "all": first window where weighted_hit == 1
      - "threshold": first window where weighted_hit >= threshold (see below)
    """
    attacked_sensors = set(get_attack_sensors(attack_sensors_dataset).get(key, []))
    N = len(ranked_features_per_window)

    if N == 0:
        return {
            "first_hit_index": -1,
            "early_score": 0.0,
            "auc_early": 0.0,
            "N_windows": 0
        }

    threshold = 0.5  # used only when mode == "threshold"

    weighted_hits = []
    first_hit_index: Optional[int] = None

    for i, ranked_features in enumerate(ranked_features_per_window):
        wh = weighted_top_k_hit_from_ranked_features(ranked_features, attacked_sensors, k)
        weighted_hits.append(wh)

        if first_hit_index is None:
            if mode == "any" and wh > 0:
                first_hit_index = i
            elif mode == "all" and wh >= 1.0:
                first_hit_index = i
            elif mode == "threshold" and wh >= threshold:
                first_hit_index = i

    # EarlyScore normalized to [0, 1]
    if first_hit_index is None:
        early_score = 0.0
        first_hit_index_out = -1
    else:
        first_hit_index_out = first_hit_index
        if N == 1:
            early_score = 1.0
        else:
            early_score = 1.0 - (first_hit_index / (N - 1))

    # AUC-style early score (average weighted hit over windows)
    auc_early = sum(weighted_hits) / N

    # Combined metrics (per-attack)
    gamma = 1.0          # choose (e.g., 1 or 2)
    eps = 1e-12          # avoid divide-by-zero
    gated_score = early_score * (auc_early ** gamma)
    harmonic_mean = (2.0 * early_score * auc_early) / (early_score + auc_early + eps)

    return {
        "first_hit_index": first_hit_index_out,
        "early_score": early_score,
        "auc_early": auc_early,
        "gated_score": gated_score,
        "harmonic_mean": harmonic_mean,
        "N_windows": N
    }


def aggregate_early_metrics(
    all_ranked_features_per_window: Dict[str, List[List[Tuple[str, float]]]],
    k: int = 5,
    mode: str = "any",
    attack_sensors_dataset: str = "swat",
) -> Dict[str, float]:
    early_scores = []
    auc_scores = []
    gated_scores = []
    harmonic_means = []
    per_key = {}

    for key, per_window in all_ranked_features_per_window.items():
        m = early_root_cause_metrics_for_key(
            per_window, key, k=k, mode=mode,
            attack_sensors_dataset=attack_sensors_dataset,
        )

        per_key[key] = {
            "first_hit_index": m["first_hit_index"],
            "early_score": m["early_score"],
            "auc_early": m["auc_early"],
            "gated_score": m["gated_score"],
            "harmonic_mean": m["harmonic_mean"],
            "N_windows": m["N_windows"],
        }

        early_scores.append(m["early_score"])
        auc_scores.append(m["auc_early"])
        gated_scores.append(m["gated_score"])
        harmonic_means.append(m["harmonic_mean"])


    return {
        "avg_early_score": sum(early_scores) / len(early_scores) if early_scores else 0.0,
        "avg_auc_early": sum(auc_scores) / len(auc_scores) if auc_scores else 0.0,
        "avg_gated_score": sum(gated_scores) / len(gated_scores) if gated_scores else 0.0,
        "avg_harmonic_mean": sum(harmonic_means) / len(harmonic_means) if harmonic_means else 0.0,
        "num_attacks": len(early_scores),
        "early_per_key": per_key,
    }




def print_metric_results(results, k):
    per_key = results[f"Top_K{k}_Per_Key"]
    early_per_key = results.get("early_per_key", {})
    headers = [
        "Attack_#", f"K{k}_Recall", f"K{k}_CWRCS", "Early_Score", "AUC_Early",
        "Gated_Score", f"K{k}_TempHM", "first_hit", "Top_K_Sensors", "Attacked_Sensors",
    ]
    rows = []
    for key, s in per_key.items():
        e = early_per_key.get(key, {})
        rows.append([
            str(key), f"{s[f'Top_K{k}_Recall']:.3f}",
            f"{s[f'Top_K{k}_Recall_Confidence']:.6f}",
            f"{e.get('early_score', float('nan')):.4f}",
            f"{e.get('auc_early', float('nan')):.4f}",
            f"{e.get('gated_score', float('nan')):.4f}",
            f"{e.get('harmonic_mean', float('nan')):.4f}",
            str(e.get('first_hit_index', -1)),
            ", ".join(s.get("top_k_sensors", [])),
            ", ".join(s.get("attacked_sensors", [])),
        ])

    widths = [max([len(header), *(len(row[i]) for row in rows)]) for i, header in enumerate(headers)]

    def format_row(row):
        return " | ".join(
            value.rjust(widths[i]) if i < 8 else value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))

    summary = [
        (f"K{k} Recall", results[f"Top_K{k}_Recall"]),
        (f"K{k} CWRCS", results[f"Top_K{k}_Recall_Confidence"]),
        ("Early Score", results.get("avg_early_score")),
        ("AUC Early", results.get("avg_auc_early")),
        ("Gated Score", results.get("avg_gated_score")),
        (f"K{k} TempHM", results.get("avg_harmonic_mean")),
    ]
    print("\nAverages: " + " | ".join(
        f"{label}: {value:.4f}" for label, value in summary if value is not None
    ))
    if "num_attacks" in results:
        print(f"Attacks evaluated: {results['num_attacks']}")



###########DEBUG CODE GIVEN BELOW##########

# # attacked_sensors = ['P102', ' AIT201', 'P101']         #['LIT101']     #["FIT401", "P501", "UV401"]
# sample_list = [('FIT101', 0.0), ('LIT101', 0.0), (' MV101', 0.0), ('P101', 0.0), ('P102', 0.0), (' AIT201', 2.8574574), ('AIT202', 0.37709224), ('AIT203', 0.0), ('FIT201', 0.0), (' MV201', 1.507465e-07), (' P201', 0.0), (' P202', 0.0), ('P203', 3.01493e-07), (' P204', 0.0), ('P205', 0.0), ('P206', 0.0), ('DPIT301', 0.0), ('FIT301', 0.0), ('LIT301', 0.0), ('MV301', 0.0), ('MV302', 0.0), (' MV303', 0.0), ('MV304', 0.0), ('P301', 1.507465e-07), ('P302', 0.0), ('AIT401', 0.035614464), ('AIT402', 0.27943563), ('FIT401', 22.613773), ('LIT401', 0.040677134), ('P401', 0.0), ('P402', 0.0), ('P403', 0.0), ('P404', 0.0), ('UV401', 17.09393), ('AIT501', 0.19161053), ('AIT502', 0.17253208), ('AIT503', 0.0), ('AIT504', 0.00816428), ('FIT501', 0.06945178), ('FIT502', 2.0656972), ('FIT503', 3.2155952), ('FIT504', 21.484236), ('P501', 22.601166), ('P502', 0.0), ('PIT501', 2.2691553), ('PIT502', 2.9677742), ('PIT503', 1.6546451), ('FIT601', 0.001992772), ('P601', 0.0), ('P602', 3.3917962e-07), ('P603', 2.2611974e-07)]
# ranked_features = sorted(
#     sample_list,
#     key=lambda x: x[1],
#     reverse=True
# )
# score_22 = compute_topk_score_for_key(
#     ranked_features=ranked_features,
#     key="22",
#     k=5
# )
# print("Top-5 sensors:")
# for s in ranked_features[:5]:
#     print(s)

# print("\nScores:", score_22)

# print('Debugging code finished :)')
