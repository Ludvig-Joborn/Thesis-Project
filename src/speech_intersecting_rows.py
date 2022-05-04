from pathlib import Path
import pandas as pd
import numpy as np

p_desed_real = Path(
    "E:/Datasets/DESED_REAL_DOWNLOAD/DESED/real/metadata/validation/validation.tsv"
)
PATH_TO_SYNTH_TRAIN_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/train/synhtetic21_train/soundscapes.tsv"
)
PATH_TO_SYNTH_VAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/validation/synhtetic21_validation/soundscapes.tsv"
)
PATH_TO_PUBLIC_EVAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/metadata/eval/public.tsv"
)
annotations = pd.read_table(p_desed_real)


# Unique filenames
a_ = len(annotations["filename"].unique())

# Get all rows with speech
df_speech = annotations.loc[annotations["event_label"] == "Speech"]
sum_speech = 0.0

for file in list(annotations["filename"].unique()):
    df_speech_filename = df_speech.loc[df_speech["filename"] == file]
    # df_speech_filename.apply(np.sum, axis=1)
    # sum_speech += row["offset"] - row["onset"]
    for i, rtm in df_speech_filename.iterrows():
        for j, row in df_speech_filename.iterrows():
            if i == j:
                continue
            if (rtm["onset"] <= row["onset"] and row["onset"] <= rtm["offset"]) or (
                rtm["onset"] <= row["offset"] and row["offset"] <= rtm["offset"]
            ):
                i_ = f"filename: {rtm['filename']} | onset: {rtm['onset']} | offset: {rtm['offset']}"
                j_ = f"filename: {row['filename']} | onset: {row['onset']} | offset: {row['offset']}"
                print(f"Row-intersection between \n{i_} \nand \n{j_}\n")


# filename	                        onset	offset	event_label
# Y0eh_N-cmcuI_350.000_360.000.wav	5.418	5.863	Speech
# Y0eh_N-cmcuI_350.000_360.000.wav	7.790	10.00	Speech
# Y0eh_N-cmcuI_350.000_360.000.wav	1.245	2.376	Speech
