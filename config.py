from src.models import (
    DeepIceModel,
    EncoderWithDirectionReconstructionV22,
    EncoderWithDirectionReconstructionV23,
)
from src.loss import loss, loss_vms


class CONFIG:
    SELECTION = "total"
    OUT = "BASELINE"
    PATH = "data/"
    NUM_WORKERS = 8
    SEED = 2023
    BS = 1024 * 3
    BS_VALID = 1024 * 3
    L = 192
    L_VALID = 192
    EPOCHS = 8
    MODEL = DeepIceModel
    MODEL_KWARGS = {"dim": 384, "dim_base": 128, "depth": 8, "head_size": 32}
    WEITHS = False
    LOSS_FUNC = loss_vms
    METRIC = loss
