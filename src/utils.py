import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from psds_eval import PSDSEval
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Tuple

# User defined imports
import config
from models.model_utils import activation


def get_datetime() -> str:
    """
    Returns the current date and time as a string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def create_path(
    dir: Path, filename: str = get_datetime(), ending: str = ".pt", best: bool = False
) -> Path:
    # Create directory
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    b = "_best" if best else ""
    path = dir / f"{filename}{b}{ending}"
    return path


def get_detection_table(
    output_table: torch.Tensor,
    file_ids: torch.Tensor,
    dataset,
) -> Dict[float, pd.DataFrame]:
    """
    Used when calculating PSD-Score.
    Calculates detection (speech) intervals on the CPU and returns a dictionary
    with detections for each operating point.
    """
    # Dictionary containing detections (pandas DataFrames)
    op_tables = {}

    # Send to CPU
    output_table = output_table.to(device="cpu", non_blocking=True)
    file_ids = file_ids.tolist()  # Creates a list on cpu

    # Iterate over operating points to add predictions to each operating point table
    for op in tqdm(config.OPERATING_POINTS, desc="Generating detection intervals"):
        detections = []
        for i, out_row in enumerate(output_table):
            out_act = activation(out_row, op)
            filename = dataset.filename(
                int(file_ids[i])
            )  # Get filename from file index
            detection_intervals = frames_to_intervals(out_act, filename)
            for speech_row in detection_intervals:
                detections.append(speech_row)

        # Add detections (as pandas DataFrame) to op_tables
        cols = ["event_label", "onset", "offset", "filename"]
        op_tables[op] = pd.DataFrame(detections, columns=cols)

    return op_tables


def get_detections(
    outputs: torch.tensor,
    file_ids: torch.Tensor,
    activation_threshold: float,
) -> List[List[Union[str, float, float, str]]]:
    """
    Used to get predictions.
    Creates and returns a list with detections (speech) intervals on the CPU for a
    given operating point.
    """
    detections = []
    for i, out_row in enumerate(outputs):
        out_act = activation(out_row, activation_threshold)
        filename = str(int(file_ids[i]))  # Filename is an index
        detection_intervals = frames_to_intervals(out_act, filename)
        for speech_row in detection_intervals:
            detections.append(speech_row)
    return detections


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
    intervals = np.linspace(0, config.CLIP_LEN_SECONDS, len(input))
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


def join_predictions(
    preds: List[List[Union[str, float, float, str]]], dataset
) -> List[List[Union[str, float, float, str]]]:
    """
    Modifies input predictions such concurrent rows with the same filename
    have their time-intervals specified in terms of the original file's timestamps.
    Also joins input predictions with overlapping boundaries.

    Example::
    ```
        [["Speech", 7.450, 10., "some_filename"], ["Speech", 0., 2.312, "some_filename"]]
            -->   [["Speech", 7.450, 12.312, "some_filename"]]
    ```
    """
    preds_updated = []
    # Loop all predictions
    for i in range(len(preds)):
        curr_row = preds[i]
        filename, file_id, seg_id = dataset.get_file_seg_ids(int(curr_row[3]))
        new_row = [  # Modify curr_row
            curr_row[0],
            curr_row[1] + config.CLIP_LEN_SECONDS * seg_id,
            curr_row[2] + config.CLIP_LEN_SECONDS * seg_id,
            filename,
        ]
        preds_updated.append(new_row)

    preds_joined = []
    i = 0
    while i < len(preds_updated):
        # Join predictions with same filename and overlapping end+start timestamps.
        curr_row = preds_updated[i]
        while (
            i + 1 < len(preds_updated)
            and preds_updated[i + 1][1] - curr_row[2]
            < config.MIN_DETECTION_INTERVAL_SEC
        ):  # Check if current row ends where next starts; If so, join them.
            curr_row = [
                curr_row[0],
                curr_row[1],
                preds_updated[i + 1][2],
                curr_row[3],
            ]
            i += 1  # Skip next row.
        preds_joined.append(curr_row)
        i += 1
    return preds_joined


def preds_to_tsv(
    preds: pd.DataFrame,
    preds_dir: str,
    path_to_audio: Path,
):
    """
    Save predictions to tsv file.
    """
    audio_dir = path_to_audio.parts[-1]
    filename = f"preds_{audio_dir}_{get_datetime()}"
    path = create_path(dir=Path(preds_dir), filename=filename, ending=".tsv")
    preds.to_csv(path, index=False, sep="\t")


def psd_score(
    op_tables: Dict[float, pd.DataFrame], annotations: pd.DataFrame
) -> Tuple[object, pd.DataFrame]:
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
    meta_table["duration"] = config.CLIP_LEN_SECONDS

    # Create PSDSEval object
    psds_eval = PSDSEval(
        **config.PSDS_PARAMS,
        ground_truth=gt_table,
        metadata=meta_table,
    )

    psds_eval.clear_all_operating_points()

    # Add operating points to PSDSEval object
    for i, op in enumerate(config.OPERATING_POINTS):
        info = {"name": f"Op {i + 1}", "threshold": op}
        psds_eval.add_operating_point(op_tables[op], info=info)

    # Get results
    psds = psds_eval.psds(
        config.PSDS_PARAMS["alpha_ct"],
        config.PSDS_PARAMS["alpha_st"],
        config.PSDS_PARAMS["max_efpr"],
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
        cc, alpha_ct=config.PSDS_PARAMS["alpha_ct"]
    )

    return psds, op_with_highest_fscore
