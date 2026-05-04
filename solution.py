"""
Main pipeline for hallucination detection.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
OUTPUT_FILE = "results.json"
PREDICTIONS_FILE = "predictions.csv"

BATCH_SIZE = 4
USE_GEOMETRIC = True

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"


def _extract_features(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    features: list[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc, unit="batch"):
        batch_texts = texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()
        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=USE_GEOMETRIC,
            )
            features.append(feat.cpu())

    return np.vstack([f.numpy() for f in features])


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Data: {DATA_FILE}")
    print(f"Max length: {MAX_LENGTH}")

    df = pd.read_csv(DATA_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(label)) for label in df["label"]], dtype=int)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    X = _extract_features(texts, model, tokenizer, device, "Extracting & aggregating")
    extract_time = time.time() - t0

    print(f"Feature matrix: {X.shape}")
    print(f"Extract time: {extract_time:.1f}s")

    splits = split_data(y, df)
    fold_results = run_evaluation(splits, X, y, HallucinationProbe)

    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    df_test = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    test_ids = df_test.index
    X_test = _extract_features(test_texts, model, tokenizer, device, "Test extraction & aggregation")

    final_probe = HallucinationProbe()
    final_probe.fit(X, y)
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)
