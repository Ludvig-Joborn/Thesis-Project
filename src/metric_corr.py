import config
import pandas as pd
from tqdm import tqdm

from models.model_utils import load_model
from eval import psd_score


def metrics_correlation(
    picked_models,
    DS_train_basepath,
    DS_val,
    log,
):
    device = "cuda"

    # picked_models = [
    #     Model("baseline", baseline),
    #     Model("improved_baseline", b_ks33_l22_gru_2),
    # ]

    sample_rates = set()
    sample_rates.add(DS_val.get_sample_rate())

    model_cumulative = pd.DataFrame()
    for picked_model in tqdm(
        iterable=picked_models,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        model = picked_model.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )

        values = {}
        epochs = config.EPOCHS
        operating_points = config.OPERATING_POINTS
        used_threshold = min(
            operating_points,
            key=lambda input_list: abs(input_list - config.ACT_THRESHOLD),
        )
        val_model_basepath = (
            DS_train_basepath / str(picked_model) / "validation" / str(DS_val)
        )

        psd_scores1 = []
        psd_scores2 = []
        fscores_epochs1 = []
        fscores_epochs2 = []
        for e in tqdm(
            iterable=range(1, epochs + 1),
            desc="Epoch",
            leave=False,
            position=1,
            colour=config.TQDM_EPOCHS,
        ):
            state = load_model(val_model_basepath / f"Ve{e}.pt")
            op_table = state["op_table"]
            psds1, fscores1 = psd_score(
                op_table,
                DS_val.get_annotations(),
                config.PSDS_PARAMS_01,
                operating_points,
            )
            psds2, fscores2 = psd_score(
                op_table,
                DS_val.get_annotations(),
                config.PSDS_PARAMS_07,
                operating_points,
            )
            try:
                fscores_epochs1.append(fscores1.Fscores.loc[used_threshold])
            except:
                fscores_epochs1.append(0)
            try:
                fscores_epochs2.append(fscores2.Fscores.loc[used_threshold])
            except:
                fscores_epochs2.append(0)

            psd_scores1.append(psds1.value)
            psd_scores2.append(psds2.value)

        values["tr_epoch_accs"] = state["tr_epoch_accs"][0:epochs]
        values["tr_epoch_losses"] = state["tr_epoch_losses"][0:epochs]
        values["val_acc_table"] = state["val_acc_table"][used_threshold][0:epochs]
        values["val_epoch_losses"] = state["val_epoch_losses"][0:epochs]
        values["Fscores1"] = fscores_epochs1
        values["psds1"] = psd_scores1
        values["Fscores2"] = fscores_epochs2
        values["psds2"] = psd_scores2

        values_df = pd.DataFrame.from_dict(values)
        tqdm.write(values_df.corr(method="pearson").to_latex())

        model_cumulative = pd.concat([model_cumulative, values_df])

    tqdm.write(model_cumulative.corr(method="pearson").to_latex())
