import polars as pl
import pickle
import os
from tqdm import tqdm
import sys
import argparse
import json


def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return config_data


def run(config):
    Nevents = {}
    for fname in tqdm(sorted(os.listdir(os.path.join(config["PATH"], "train")))):
        path = os.path.join(config["PATH"], "train", fname)
        df = pl.read_parquet(path)
        df = df.groupby("event_id").agg([pl.count()])

        Nevents[fname] = {}
        Nevents[fname]["total"] = len(df)
        Nevents[fname]["short"] = len(df.filter(pl.col("count") < 64))
        Nevents[fname]["medium"] = len(
            df.filter((pl.col("count") >= 64) & (pl.col("count") < 192))
        )
        Nevents[fname]["long"] = len(df.filter(pl.col("count") >= 192))

    with open(os.path.join(config["PATH"], "Nevents.pickle"), "wb") as f:
        pickle.dump(Nevents, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description="Create a model from a JSON config file."
    )
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    args = parser.parse_args()
    config_file_path = args.config_file
    config_data = read_config_file(config_file_path)
    run(config_data)
