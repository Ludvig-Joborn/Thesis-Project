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
annotations = pd.read_table(PATH_TO_PUBLIC_EVAL_DESED_TSV)


# Unique filenames
a_ = len(annotations["filename"].unique())

# Get all rows with speech
df_speech = annotations.loc[annotations["event_label"] == "Speech"]
sum_speech = 0.0
for index, row in df_speech.iterrows():
    sum_speech += row["offset"] - row["onset"]

# Prints
print(
    f"Rows of Speech: {len(df_speech)} / {len(annotations)} = {100*len(df_speech)/len(annotations)}\%"
)
print(
    f"Rows of Speech: {sum_speech/60:1f} / {a_ * 10.0/60:1f} = {100*(sum_speech/60)/(a_ * 10.0/60):1f}\%"
)
print("nr unique filenames:", a_)
# print(f"-> Total minutes in dataset: {a_ * 10.0:1f} | hours: {a_ * 10.0/60:1f}")
# print(f"-> Total minutes of speech: {sum_speech:1f} | hours: {sum_speech/60:1f}")
print(f"Speech: {sum_speech / (a_ * 10.0):2f}")
