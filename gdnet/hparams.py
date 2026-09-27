"""GD-Net hyper-parameters, as given in the paper's Supporting Information
("NOTE: Settings of Hyper-Parameters", J Cell Mol Med, Appendix S1):

    "Hidden layer 1 consists of 1024 nodes, while hidden layer 2 has 128
     nodes. The node number in middle hidden layer was chosen from the set
     [10, 20, 50]. The learning rate (LR) was chosen from the range
     [1e-2, 1e-3, 1e-4, 1e-5], and the maximum epoch for training was set
     200. The momentum coefficient was set 0.99. The parameters were
     selected based on the 5-fold results in the experiments."

This module is the single source of truth for those values; main.py,
run_pipeline.py and the extensions all import from here.

Interpretation notes:
  * "middle hidden layer" = the bottleneck / embedding layer that follows the
    1024- and 128-node hidden layers, i.e. the multi-omics meta-features fed
    to the Cox-EN model. The encoder is therefore  F -> 1024 -> 128 -> {10,20,50}.
  * "momentum coefficient" is the MoCo key-encoder momentum (lambda / m of
    paper Eq. 5), which is what that term denotes in momentum-contrast
    methods. The SGD optimiser momentum is not given in the SI and keeps the
    official repo's value (0.9).
  * "selected based on the 5-fold results" = grid search over
    LOW_DIM_GRID x LR_GRID, keeping the combination with the highest mean
    5-fold Cox-EN C-index (see gdnet/tuning.py).
"""

import types

HIDDEN1 = 1024                              # hidden layer 1
HIDDEN2 = 128                               # hidden layer 2
LOW_DIM_GRID = (10, 20, 50)                 # middle (embedding) layer candidates
LR_GRID = (1e-2, 1e-3, 1e-4, 1e-5)          # learning-rate candidates
EPOCHS = 200                                # maximum training epochs
MOCO_MOMENTUM = 0.99                        # momentum coefficient (Eq. 5)
N_FOLDS = 5                                 # folds used for model selection

# Used when no search is run (e.g. main.py, which has no survival data to
# score candidates with). Middle of the SI grid / largest SI learning rate.
DEFAULT_LOW_DIM = 20
DEFAULT_LR = 1e-2

# Not specified in the SI -> official GD-Net repo defaults.
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.2
QUEUE_SIZE = 512
BATCH_SIZE = 512


def train_args(**overrides) -> types.SimpleNamespace:
    """Namespace accepted by main.train_encoder(), pre-filled with the SI
    settings. Used by extensions that train the encoder programmatically."""
    ns = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=DEFAULT_LR,
              momentum=SGD_MOMENTUM, wd=WEIGHT_DECAY, cos=True,
              low_dim=DEFAULT_LOW_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2,
              moco_r=QUEUE_SIZE, moco_m=MOCO_MOMENTUM, temperature=TEMPERATURE,
              out=None)
    ns.update(overrides)
    return types.SimpleNamespace(**ns)
