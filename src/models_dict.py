# Baseline, basic
from models.model_extensions_1.basic_nn import NeuralNetwork as basic_nn
from models.baseline import NeuralNetwork as baseline
from models.improved_baseline import NeuralNetwork as improved_baseline

# b2
from models.model_extensions_1.b2 import NeuralNetwork as b2
from models.model_extensions_1.b2_cbam import NeuralNetwork as b2_cbam
from models.model_extensions_1.b2_cbam_drop01 import NeuralNetwork as b2_cbam_drop01
from models.model_extensions_1.b2_cbam_drop01_lindrop import (
    NeuralNetwork as b2_cbam_drop01_lindrop,
)
from models.model_extensions_1.b2_cbam_drop02 import NeuralNetwork as b2_cbam_drop02

# b1
from models.model_extensions_1.b1 import NeuralNetwork as b1
from models.model_extensions_1.b1_cbam import NeuralNetwork as b1_cbam
from models.model_extensions_1.b1_cbam_drop01 import NeuralNetwork as b1_cbam_drop01
from models.model_extensions_1.b1_cbam_drop01_lindrop import (
    NeuralNetwork as b1_cbam_drop01_lindrop,
)
from models.model_extensions_1.b1_cbam_drop02 import NeuralNetwork as b1_cbam_drop02

# Pooling
from models.pooling_extensions.lp_pool import NeuralNetwork as lp_pool
from models.pooling_extensions.max_pool import NeuralNetwork as max_pool

# Model extensions 2
from models.model_extensions_2.baseline_ks33_l22 import (
    NeuralNetwork as baseline_ks33_l22,
)
from models.model_extensions_2.baseline_ks33_l44 import (
    NeuralNetwork as baseline_ks33_l44,
)
from models.model_extensions_2.baseline_ks53_l24 import (
    NeuralNetwork as baseline_ks53_l24,
)
from models.model_extensions_2.baseline_ks73_l12 import (
    NeuralNetwork as baseline_ks73_l12,
)

# Recurrent momory units like LSTM and GRU
from models.recurrent_memory_ext.gru_2_drop01 import NeuralNetwork as gru_2_drop01
from models.recurrent_memory_ext.gru_2 import NeuralNetwork as gru_2
from models.recurrent_memory_ext.gru_4 import NeuralNetwork as gru_4
from models.recurrent_memory_ext.lstm_1 import NeuralNetwork as lstm_1
from models.recurrent_memory_ext.lstm_2_drop01 import NeuralNetwork as lstm_2_drop01
from models.recurrent_memory_ext.lstm_3_drop01 import NeuralNetwork as lstm_3_drop01

# Activation functions
from models.act_funcs.ls_relu import NeuralNetwork as ls_relu
from models.act_funcs.ls_relu_tr import NeuralNetwork as ls_relu_tr
from models.act_funcs.swish import NeuralNetwork as swish
from models.act_funcs.swish_tr import NeuralNetwork as swish_tr

# Model Ext 3
from models.model_extensions_3.b_ks33_l22_gru_2 import NeuralNetwork as b_ks33_l22_gru_2
from models.model_extensions_3.b_ks33_l22_gru_2_relu_swish_tr import (
    NeuralNetwork as b_ks33_l22_gru_2_relu_swish_tr,
)
from models.model_extensions_3.b_ks33_l22_gru_2_swish_tr import (
    NeuralNetwork as b_ks33_l22_gru_2_swish_tr,
)


class Model:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def __str__(self) -> str:
        return self.name

    def get_NN(self):
        return self.model


MODELS = [
    # Model("basic_nn", basic_nn),
    #
    Model("baseline", baseline),
    # Model("improved_baseline", improved_baseline),
    #
    # Model("b2", b2),
    # Model("b2_cbam", b2_cbam),
    # Model("b2_cbam_drop01", b2_cbam_drop01),
    # Model("b2_cbam_drop01_lindrop", b2_cbam_drop01_lindrop),
    # Model("b2_cbam_drop02", b2_cbam_drop02),
    # Model("b1", b1),
    # Model("b1_cbam", b1_cbam),
    # Model("b1_cbam_drop01", b1_cbam_drop01),
    # Model("b1_cbam_drop01_lindrop", b1_cbam_drop01_lindrop),
    # Model("b1_cbam_drop02", b1_cbam_drop02),
    #
    Model("lp_pool", lp_pool),
    Model("max_pool", max_pool),
    Model("baseline_ks33_l22", baseline_ks33_l22),
    Model("baseline_ks33_l44", baseline_ks33_l44),
    Model("baseline_ks53_l24", baseline_ks53_l24),
    Model("baseline_ks73_l12", baseline_ks73_l12),
    Model("gru_2_drop01", gru_2_drop01),
    Model("gru_2", gru_2),
    Model("gru_4", gru_4),
    Model("lstm_1", lstm_1),
    Model("lstm_2_drop01", lstm_2_drop01),
    Model("lstm_3_drop01", lstm_3_drop01),
    Model("b_ks33_l22_gru_2", b_ks33_l22_gru_2),
    Model("b_ks33_l22_gru_2_relu_swish_tr", b_ks33_l22_gru_2_relu_swish_tr),
    Model("b_ks33_l22_gru_2_swish_tr", b_ks33_l22_gru_2_swish_tr),
    # Model("ls_relu", ls_relu),
    # Model("ls_relu_tr", ls_relu_tr),
    Model("swish", swish),
    Model("swish_tr", swish_tr),
]
