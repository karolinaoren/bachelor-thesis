# Utility functions to create and evaluate the BLAST-kNN algorithm for comparison.


import os
import subprocess
import tempfile
import numpy as np
import random


k_threshold = 10
e_val = "1"


def save_dict(file, dictionary):
    np.save(file + ".npy", dictionary)


def load_dict(file):
    return np.load(file + ".npy", allow_pickle=True).item()


def parse_fasta_to_dict(file):
    protein_dict = {}
    current_protein = None
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('>sp'):
                parts = line.split("|")
                current_protein = parts[1]
                protein_dict[current_protein] = ""
            elif line.startswith('>'):
                current_protein = line[1:]
                protein_dict[current_protein] = ""
            elif current_protein is not None:
                protein_dict[current_protein] += line
    return protein_dict


def make_blastdb(source_file, title):
    cmd = [
        "makeblastdb",
        "-in", source_file,
        "-parse_seqids",
        "-title", title,
        "-dbtype", "prot",
        "-out", title
    ]
    subprocess.call(cmd)
    print("blast db done")


def blastp(seq, prot):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_fasta:
        temp_fasta.write(f">{prot}\n{seq}")
        temp_fasta_path = temp_fasta.name
    blast_cmd = [
        "blastp",
        "-query", temp_fasta_path,
        "-out", "resources/prot/" + prot + "_out.txt",
        "-db", "cafa3",
        "-evalue", e_val,
        "-outfmt", "6 qseqid sseqid bitscore evalue",
        "-num_threads", "16"
    ]
    subprocess.call(blast_cmd)
    os.remove(temp_fasta_path)


def blast_seqs(file):
    seq_dict = parse_fasta_to_dict(file)
    keys = list(seq_dict.keys())
    random.shuffle(keys)
    num_keys = len(keys)
    shuffled_dict = {key: seq_dict[key] for key in keys[:num_keys]}

    for prot in shuffled_dict:
        blastp(seq_dict.get(prot), prot)


def get_go_terms_indices(go_terms):
    go_indices = {}
    for i, go_term in enumerate(go_terms):
        go_indices[go_term] = i
    return go_indices


def create_annotations_dict(mapping_file, go_indices):
    annotations = {}
    current_protein = None
    protein_annotations = None
    with open(mapping_file, "r") as file:
        for line in file:
            parts = line.strip().split('\t')
            protein = parts[0]
            go_term = parts[1]
            if protein != current_protein:
                if current_protein is not None:
                    annotations[current_protein] = protein_annotations
                current_protein = protein
                protein_annotations = [0] * len(go_indices)
            if go_term in go_indices:
                protein_annotations[go_indices[go_term]] = 1
            annotations[protein] = protein_annotations
    if current_protein is not None:
        annotations[current_protein] = protein_annotations
    return annotations


def get_knn(prot):
    similar_data = []
    k = 0
    with open("resources/prot/" + prot + "_out.txt", "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            query_seq = parts[0]
            similar_seq = parts[1]
            bit_score = float(parts[2])
            if similar_seq != query_seq and k <= k_threshold:
                similar_data.append((similar_seq, bit_score))
            k += 1
    return similar_data


def get_go_scores(similar_data, go_terms, annotations):
    go_scores = [0] * len(go_terms)
    total_annotated = [0] * len(go_terms)
    total_annotated_bit_scores = [0] * len(go_terms)
    total_bit_scores = [0] * len(go_terms)
    for i in range(len(go_terms)):
        for protein, bit_score in similar_data:
            if protein in annotations and annotations.get(protein)[i] == 1:
                total_annotated[i] += 1
                total_annotated_bit_scores[i] += bit_score
            total_bit_scores[i] += bit_score
        if total_annotated[i] > 0:
            go_scores[i] = total_annotated_bit_scores[i] / total_bit_scores[i]
        else:
            go_scores[i] = 0.0
    return go_scores


def get_test_data(seqs, test_size):
    keys = list(seqs.keys())
    random.shuffle(keys)
    num_keys = len(keys) // test_size
    selected_dict = {key: seqs[key] for key in keys[:num_keys]}
    return selected_dict


def blast_knn(file, mapping_file, annotation_file, prediction_file, go_terms):
    go_indices = get_go_terms_indices(go_terms)
    print(go_indices)
    annotations = create_annotations_dict(mapping_file, go_indices)
    save_dict(annotation_file, annotations)
    go_scores = {}
    seq_dict = parse_fasta_to_dict(file)
    for prot in seq_dict:
        knn = get_knn(prot)
        go_scores[prot] = get_go_scores(knn, go_terms, annotations)
    save_dict(prediction_file, go_scores)


def assign_go_terms(threshold, go_scores):
    predictions = {}
    for prot in go_scores:
        predictions[prot] = [0] * len(go_scores.get(prot))
        for i in range(len(go_scores.get(prot))):
            if go_scores.get(prot)[i] >= threshold:
                predictions.get(prot)[i] = 1
    return predictions


def get_precisions_recalls(predictions, ground_truth):
    precisions = []
    recalls = []
    for prot in predictions:
        tp, fp, fn = 0, 0, 0
        for i in range(len(predictions.get(prot))):
            if predictions.get(prot)[i] == 1:
                if prot in predictions and prot in ground_truth and predictions.get(prot)[i] == ground_truth.get(prot)[i]:
                    tp += 1
                else:
                    fp += 1
            else:
                if prot in predictions and prot in ground_truth and predictions.get(prot)[i] != ground_truth.get(prot)[i]:
                    fn += 1
        precision, recall = 0., 0.
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def evaluate(go_scores, ground_truth, result_file):
    lines = []
    max_f1 = 0.
    for t in range(10, 101, 10):
        t /= 100
        predictions = assign_go_terms(t, go_scores)
        precisions, recalls = get_precisions_recalls(predictions, ground_truth)
        precisions_mean = np.mean(precisions)
        recalls_mean = np.mean(recalls)
        f1 = 0.
        if (precisions_mean + recalls_mean) != 0:

            f1 = 2 * precisions_mean * recalls_mean / (precisions_mean + recalls_mean)
        if max_f1 < f1:
            max_f1 = f1
        print("t = " + str(t) + ", f1 = " + str(f1) + ", max_f1 = " + str(max_f1))
        lines.append("t = " + str(t) + ", f1 = " + str(f1) + ", max_f1 = " + str(max_f1))
    with open(result_file, "a") as f:
        f.write("\ne_val = " + str(e_val) + "; k = " + str(k_threshold) + "\n")
        for line in lines:
            f.write(line + "\n")

