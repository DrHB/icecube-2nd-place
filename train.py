import polars as pl
import pandas as pd
from src.fastai_fix import *
from tqdm.notebook import tqdm
from src.dataset import (
    RandomChunkSampler,
    LenMatchBatchSampler,
    IceCubeCache,
    DeviceDataLoader,
)
from src.loss import loss, loss_vms
from fastxtend.vision.all import EMACallback
from tqdm import tqdm
from src.utils import seed_everything, WrapperAdamW
import config


def train(config):
    ds_train = IceCubeCache(
        config.PATH,
        mode="train",
        L=config.L,
        selection=config.SELECTION,
        reduce_size=0.125,
    )
    ds_train_len = IceCubeCache(
        config.PATH,
        mode="train",
        L=config.L,
        reduce_size=0.125,
        selection=config.SELECTION,
        mask_only=True,
    )
    sampler_train = RandomChunkSampler(ds_train_len, chunks=ds_train.chunks)
    len_sampler_train = LenMatchBatchSampler(
        sampler_train, batch_size=config.BS, drop_last=True
    )
    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_train,
            batch_sampler=len_sampler_train,
            num_workers=config.NUM_WORKERS,
            persistent_workers=True,
        )
    )

    ds_val = IceCubeCache(
        config.PATH, mode="eval", L=config.L_VALID, selection=config.SELECTION
    )
    ds_val_len = IceCubeCache(
        config.PATH,
        mode="eval",
        L=config.L_VALID,
        selection=config.SELECTION,
        mask_only=True,
    )
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(
        sampler_val, batch_size=config.BS_VALID, drop_last=False
    )
    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_val, batch_sampler=len_sampler_val, num_workers=0
        )
    )

    data = DataLoaders(dl_train, dl_val)
    model = config.MODEL(**config.MODEL_KWARGS)
    if config.WEITHS:
        print("Loading weights from ...", config.WEITHS)
        model.load_state_dict(torch.load(config.WEITHS))
    model = nn.DataParallel(model)
    model = model.cuda()
    learn = Learner(
        data,
        model,
        path=config.OUT,
        loss_func=config.LOSS_FUNC,
        cbs=[
            GradientClip(3.0),
            CSVLogger(),
            SaveModelCallback(monitor="loss", comp=np.less, every_epoch=True),
            GradientAccumulation(n_acc=4096 // config.BS),
        ],
        metrics=[config.METRIC],
        opt_func=partial(WrapperAdamW, eps=1e-7),
    ).to_fp16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    args = parser.parse_args()
    configs = eval(f"config.{args.config_name}")
    print(f"Training with config: {configs.__dict__}")
    print("_______________________________________________________")
    seed_everything(configs.SEED)
    os.makedirs(configs.OUT, exist_ok=True)
    train(configs)
