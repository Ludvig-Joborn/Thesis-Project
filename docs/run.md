## Setup virtual environment
### Windows
```bash
# Activate the environment (bash)
source venv/bin/activate # may be 'venv/Scripts/activate'

# Activate the environment (powershell)
./venv/Scripts/Activate.ps1
```

## Configurations before running
In `config.py`, six fields must be modified before running. These are:
- `PATH_TO_SYNTH_TRAIN_DESED_TSV` - path to training data labels.
- `PATH_TO_SYNTH_TRAIN_DESED_WAVS` - path to training data wavs (.wav audio files).
- `PATH_TO_SYNTH_VAL_DESED_TSV` - path to validation data labels.
- `PATH_TO_SYNTH_VAL_DESED_WAVS` - path to validation data wavs.
- `PATH_TO_PUBLIC_EVAL_DESED_TSV` - path to testing data labels.
- `PATH_TO_PUBLIC_EVAL_DESED_WAVS` - path to testing data wavs.
- `SAVED_MODEL_DIR` - path to where models are saved.

Additionally, if you desire to run the models to produce predictions on your own files, `PATH_TO_CUSTOM_WAVS` needs to be modified to the path of directory where the files are contained.

Example:
```python
PATH_TO_SYNTH_TRAIN_DESED_TSV = Path("E:/Datasets/dcase21_synth/metadata/train/synhtetic21_train/soundscapes.tsv")
```

## Running the program
To run the program, use the following command:
```python
.\src\run.py [-flags]
```
The command requires at least one of the following flags:
- `-tr` - runs training of models (`-ftr` forces model retraining).
- `-val` - runs validation.
- `-te_acc_loss` - runs testing, calculating accuracy and loss.
- `-te_psds` - runs testing, calculating PSDS (Polyphonic Sound Detection Score).
- `-pred` - creates predictions for non-dataset files.
- `-plot` - plots metrics for all epochs of training. The content of the plot is specified by the argument `WHAT_TO_PLOT` in `config.py`.

The flags can be combined. Example:
```python
.\src\run.py -tr -val -te_psds -plot
```
This will run training, validation, testing for PSDS, and create plots.

For more information on what parameters are available and their usage, use the `--help` flag.

## Selecting models to run
After starting `run.py`, you will be asked to select what model(s) to use. The program will run through all models before moving to the next step, e.g. train all models before validating. If you wish to add or remove models to the list, modify `src/models_dict.py`.