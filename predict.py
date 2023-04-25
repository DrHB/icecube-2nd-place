import polars as pl
import pandas as pd
from tqdm.notebook import tqdm
from src.data_test import (
    IceCubeDataset,
    get_val,
    dict_to,
    LenMatchBatchSampler,
    DataLoader,
)
from tqdm import tqdm
from src.utils import seed_everything
import argparse
import json
import torch
import os
from src.models import (
    DeepIceModel,
    EncoderWithDirectionReconstructionV22,
    EncoderWithDirectionReconstructionV23,
)
from pdb import set_trace
from functools import partial


def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return config_data


def main():
    parser = argparse.ArgumentParser(
        description="Create a model from a JSON config file."
    )
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    parser.add_argument(
        "configs",
        nargs="*",
        metavar=("KEY", "VALUE"),
        help="The JSON config key to override and its new value.",
    )

    args = parser.parse_args()
    config_file_path = args.config_file

    config_data = read_config_file(config_file_path)

    if args.configs:
        for config_key, config_value in zip(args.configs[::2], args.configs[1::2]):
            keys = config_key.split(".")
            last_key = keys.pop()

            current_data = config_data
            for key in keys:
                current_data = current_data[key]

            try:
                value = json.loads(config_value)
            except json.JSONDecodeError:
                value = config_value

            current_data[last_key] = value

    MODELS = [
        [
            "ice-cube-final-models/baselineV3_BE_globalrel_d32_0_6ema.pth",
            partial(DeepIceModel, dim=768, dim_base=192, depth=12, head_size=32),
            0.08254897,
        ],
        [
            "ice-cube-final-models/baselineV3_BE_globalrel_d64_0_3emaFT_2.pth",
            partial(DeepIceModel, dim=768, dim_base=192, depth=12, head_size=64),
            0.15350807,
        ],
        [
            "ice-cube-final-models/VFTV3_4RELFT_7.pth",
            partial(
                DeepIceModel, dim=768, dim_base=192, depth=12, head_size=32, n_rel=4
            ),
            0.19367443,
        ],
        [
            "ice-cube-final-models/V22FT6_1.pth",
            partial(
                EncoderWithDirectionReconstructionV22,
                dim=384,
                dim_base=128,
                depth=8,
                head_size=32,
            ),
            0.23597202,
        ],
        [
            "ice-cube-final-models/V23FT5_6.pth",
            partial(
                EncoderWithDirectionReconstructionV23,
                dim=768,
                dim_base=192,
                depth=12,
                head_size=64,
            ),
            0.3342965,
        ],
    ]
    seed_everything(config_data["SEED"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ICE_PROPERTIES = config_data["PATH"]

    ds = IceCubeDataset(config_data["PATH"], ICE_PROPERTIES, L=config_data["L"])
    len_sampler = LenMatchBatchSampler(
        torch.utils.data.SequentialSampler(ds),
        batch_size=config_data["BS"],
        drop_last=False,
    )
    dl = DataLoader(ds, batch_sampler=len_sampler, num_workers=0)

    models, weights = [], []
    for path, Model, w in MODELS:
        print("loading:", path)
        model = Model()
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        model.to(device)
        models.append(model)
        weights.append(w)
    weights = torch.FloatTensor(weights)
    weights /= weights.sum()

    preds = []
    for x in tqdm(dl):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                x = dict_to(x, device)
                p = (
                    torch.stack(
                        [
                            torch.nan_to_num(model(x)).clip(-1000, 1000)
                            for model in models
                        ],
                        -1,
                    ).cpu()
                    * weights
                ).sum(-1)
        p = get_val(p).numpy()
        for pi, idx in zip(p, x["idx"]):
            preds.append(
                {"event_id": idx.cpu().item(), "azimuth": pi[0], "zenith": pi[1]}
            )

    df = pd.read_parquet(os.path.join(config_data["PATH"], "sample_submission.parquet"))
    df = pd.merge(
        df["event_id"], pd.DataFrame(preds), on="event_id", how="left"
    ).fillna(value=0)
    df.to_csv("submission.csv", index=False)
    df.head()


if __name__ == "__main__":
    main()
