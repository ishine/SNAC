import math
import multiprocessing
import os
import argparse
from pathlib import Path
from random import shuffle

import torch
from glob import glob
from tqdm import tqdm

import wav2ssl
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np
ssl_type = "chinese_hubert"


def process_one(file_path, model):
    path = Path(file_path)
    name = path.stem

    spk_name = path.parent.name

    ssl_path = f"dataset/{spk_name}/{name}.ssl.pt"

    if not os.path.exists(ssl_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k, sr = librosa.load(path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = wav2ssl.get_ssl_content(ssl_type, model, wav_16k_tensor=wav16k)
        torch.save(ssl_content.cpu(), ssl_path)
        print(ssl_path+"!!!!!!")
        exit(0)


def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl_model = wav2ssl.get_ssl_model(ssl_type).to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()
