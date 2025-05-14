# Utility functions to create embeddings with ESM-2 and evaluate using Logistic Regression classifier for comparison.


import numpy as np
import pandas as pd
from esm import FastaBatchedDataset, pretrained
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch=4096, seq_length=1022, repr_layers=[6]):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                entry_id = label.split()[0]

                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                    layer: t[i, 1: truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                print(f'{entry_id} done')
                torch.save(result, filename)


def get_protein_ids(pickle_file):
    df = pd.read_pickle(pickle_file)
    return df['protein_id'].tolist()


def load_embeddings(embedding_dir, protein_ids):
    embeddings = {}
    for protein_id in protein_ids:
        embedding = torch.load(f'{embedding_dir}/{protein_id}.pt')
        embeddings[protein_id] = embedding['mean_representations'][6].numpy()
    return embeddings


def load_labels(label_file, unique_go_terms):
    df = pd.read_pickle(label_file)
    df['go_terms'] = df['go_terms'].apply(lambda x: x.split(', '))

    labels = {}
    for _, row in df.iterrows():
        protein_id = row['protein_id']
        go_terms = row['go_terms']
        binary_array = binarize_go_terms(go_terms, unique_go_terms)
        labels[protein_id] = binary_array

    return labels


def binarize_go_terms(go_terms, unique_go_terms):
    binary_array = np.zeros(len(unique_go_terms), dtype=int)
    for term in go_terms:
        if term in unique_go_terms:
            index = unique_go_terms.index(term)
            binary_array[index] = 1
    return binary_array


def train_classifier(X_train, y_train):
    classifier = MultiOutputClassifier(LogisticRegression())

    try:
        classifier.fit(X_train, y_train)
    except Exception as e:
        print(f"An error occurred while fitting the classifier: {e}")
        return None
    return classifier
