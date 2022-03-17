import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from psds_eval import PSDSEval, plot_psd_roc
from typing import List, Dict, Union, Tuple

# User defined imports
from config import *
from models.model_utils import activation


def get_detection_table(
    output_table: torch.Tensor,
    file_ids: torch.Tensor,
    dataset,
) -> Dict[float, pd.DataFrame]:
    """
    Calculates detection (speech) intervals on the CPU and returns a dictionary
    with detections for each operating point.
    """
    # Dictionary containing detections (pandas DataFrames)
    op_tables = {}

    # Send to CPU
    output_table = output_table.to(device="cpu", non_blocking=True)
    file_ids = file_ids.tolist()  # Creates a list on cpu

    # Iterate over operating points to add predictions to each operating point table
    for op in tqdm(OPERATING_POINTS, desc="Generating detection intervals"):
        detections = []
        for i, out_row in enumerate(output_table):
            out_act = activation(out_row, op)
            # Get filename from file index
            filename = dataset.filename(int(file_ids[i]))
            detection_intervals = frames_to_intervals(out_act, filename)
            for speech_row in detection_intervals:
                detections.append(speech_row)

        # Add detections (as pandas DataFrame) to op_tables
        cols = ["event_label", "onset", "offset", "filename"]
        op_tables[op] = pd.DataFrame(detections, columns=cols)

    return op_tables


def frames_to_intervals(
    input: torch.Tensor, filename: str
) -> List[List[Union[str, float, float, str]]]:
    """
    Converts frame-wise output into time-intervals
    for each row where speech is detected.
    The return format is the following::

    ```
        [["event_label", "onset", "offset", "filename"], ...]
    ```
    For example::

    ```
    [
        ["Speech", 0.2, 5.6, "audio_file1.wav"],  # Speech detection 1
        ["Speech", 7.3, 10.0, "audio_file2.wav"],  # Speech detection 2
        ...
    ]
    ```
    """
    outputs = []
    activity = []
    detecting_speech = False
    intervals = np.linspace(0, CLIP_LEN_SECONDS, len(input))
    for i in range(len(intervals)):
        if input[i] == 1:
            if not detecting_speech:
                detecting_speech = True
                activity.append("Speech")
                activity.append(intervals[i])  # Start speech interval
        if input[i] == 0:
            if detecting_speech:
                detecting_speech = False
                activity.append(intervals[i - 1])  # End speech interval
                activity.append(filename)
                outputs.append(activity)
                activity = []

    if detecting_speech:  # Catch cases where the final frame contains speech.
        activity.append(intervals[i])
        activity.append(filename)
        outputs.append(activity)

    return outputs


def psd_score(
    op_tables: Dict[float, pd.DataFrame], annotations: pd.DataFrame, plot: bool = False
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate PSDS Score, F1-Score and plot a ROC-PSD curve if specified.

    The input-dictionary 'op_tables' must contain an operating point (key),
    and a pandas DataFrame with detections (value).

    The format of the detections must have the following format::

    ```
    columns = ["event_label", "onset", "offset", "filename"]
       (type: [str,           float,   float,    str       ])
    ```
    """
    # Ground Truth table
    gt_table = annotations

    # Keep only speech
    gt_table = gt_table.loc[gt_table["event_label"] == "Speech"]
    gt_table = gt_table.reset_index(drop=True)

    # Meta data neccessary for PSDS
    meta_table = pd.DataFrame(gt_table.loc[:, "filename"])
    meta_table["duration"] = CLIP_LEN_SECONDS

    # Create PSDSEval object
    psds_eval = PSDSEval(
        **PSDS_PARAMS,
        ground_truth=gt_table,
        metadata=meta_table,
    )

    psds_eval.clear_all_operating_points()

    # Add operating points to PSDSEval object
    for i, op in enumerate(OPERATING_POINTS):
        info = {"name": f"Op {i + 1}", "threshold": op}
        psds_eval.add_operating_point(op_tables[op], info=info)

    # Get results
    psds = psds_eval.psds(
        PSDS_PARAMS["alpha_ct"],
        PSDS_PARAMS["alpha_st"],
        PSDS_PARAMS["max_efpr"],
    )

    # Get F1-Score
    cc = pd.DataFrame(
        [
            {
                "class_name": psds_eval.class_names[0],
                "constraint": "fscore",
                "value": None,
            }
        ]
    )
    op_with_highest_fscore = psds_eval.select_operating_points_per_class(
        cc, alpha_ct=PSDS_PARAMS["alpha_ct"]
    )

    if plot:
        # Plot the PSD-ROC
        plt.style.use("fast")
        plot_psd_roc(psds)

    return psds.value, op_with_highest_fscore
