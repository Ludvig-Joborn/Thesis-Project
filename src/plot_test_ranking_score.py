import matplotlib.pyplot as plt
import warnings
import collections
import config
from models_dict import *
from models.model_utils import *
from eval import *
from utils import timer
import time
from collections import OrderedDict


# Ignore warnings
# warnings.simplefilter("ignore", UserWarning)


# DESED Public Eval
D_EVAL = OrderedDict(
    {
        "baseline": 0.773873,
        "improved baseline": 0.677182,
        "baseline SNR 30": 0.783914,
        "improved baseline SNR 30": 0.613346,
        "baseline SNR 20": 0.773621,
        "improved baseline SNR 20": 0.604109,
        "baseline SNR 15": 0.764076,
        "improved baseline SNR 15": 0.548803,
        "baseline SNR 10": 0.746578,
        "improved baseline SNR 10": 0.586690,
        "baseline SNR 5": 0.432400,
        "improved baseline SNR 5": 0.546781,
        "baseline SNR 0": 0.733399,
        "improved baseline SNR 0": 0.623749,
        "baseline SNR -5": 0.377352,
        "improved baseline SNR -5": 0.654660,
        "baseline SNR -10": 0.290611,
        "improved baseline SNR -10": 0.845532,
    }
)

# DESED Real
D_REAL = OrderedDict(
    {
        "baseline": 0.751035,
        "improved baseline": 0.582087,
        "baseline SNR 30": 0.676503,
        "improved baseline SNR 30": 0.519653,
        "baseline SNR 20": 0.690416,
        "improved baseline SNR 20": 0.515577,
        "baseline SNR 15": 0.715606,
        "improved baseline SNR 15": 0.470616,
        "baseline SNR 10": 0.697498,
        "improved baseline SNR 10": 0.565027,
        "baseline SNR 5": 0.405564,
        "improved baseline SNR 5": 0.494728,
        "baseline SNR 0": 0.648763,
        "improved baseline SNR 0": 0.519937,
        "baseline SNR -5": 0.424748,
        "improved baseline SNR -5": 0.594333,
        "baseline SNR -10": 0.200537,
        "improved baseline SNR -10": 0.703042,
    }
)

SNRS = ["Unmodified", 30, 20, 15, 10, 5, 0, -5, -10]


def plot_models(D_, title):
    # Parameters for plotting
    plot_params = {
        "linestyle": "-",
        "marker": "o",
        "markersize": 4,
    }
    plt.style.use("ggplot")
    plt.figure(title, figsize=(10, 8))

    x = SNRS
    vals = list(D_.values())
    # labels = list(D_.keys())
    plt.plot(x, vals[0::2], **plot_params, label="Baseline")
    plt.plot(x, vals[1::2], **plot_params, label="Improved Baseline")
    plt.ylabel("Ranking Score")
    plt.xlabel("Signal-to-noise ratio [dB]")
    plt.title(f"Testdataset: {title} - Ranking Score")
    plt.legend()
    plt.ylim(0, 1)

    plt.show()


plot_models(D_EVAL, "DESED Public Eval")

plot_models(D_REAL, "DESED Real")
