# Utility functions to evaluate results and combine them into a table.


import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import nn


def auroc_to_tsv(result_path, root_go, term_file_reduced, term_file, output_file, iter_idx, from_level):
    df = pd.read_csv(term_file_reduced)
    go_terms = df['Term'].tolist()

    with open(output_file, 'w') as f:
        f.write("GO_Term,AUROC,From_Level\n")

        for go in go_terms:
            if go == root_go:
                continue

            i = nn.get_index_by_term(term_file, go)

            true_labels_df = pd.read_pickle(f'resources/{nn.dom}/no_tl/test/test_{go[3:]}_{iter_idx}.pkl')
            true_labels = np.array(true_labels_df['go_terms'].tolist())
            true_labels = true_labels[:, i]

            try:
                predictions_df = pd.read_pickle(f'{result_path}/predictions/predictions_{go[3:]}_{iter_idx}.pkl')
                predictions = np.array(predictions_df['predictions'].tolist())

                try:
                    auroc = roc_auc_score(true_labels, predictions)
                    if not from_level:
                        f.write(f"{go},{auroc:.4f},NaN\n")
                    else:
                        f.write(f"{go},{auroc:.4f},{from_level+1}\n")
                except ValueError:
                    f.write(f"{go},NaN\n")

            except:
                print(f'{go} not found in preds')


def f1_to_tsv(result_path, root_go, term_file_reduced, term_file, output_file, iter_idx, threshold):
    df = pd.read_csv(term_file_reduced)
    go_terms = df['Term'].tolist()

    with open(output_file, 'w') as f:
        f.write("GO_Term,Precision,Recall,F1\n")

        for go in go_terms:
            if go == root_go:
                continue

            i = nn.get_index_by_term(term_file, go)

            true_labels_df = pd.read_pickle(f'resources/{nn.dom}/no_tl/test/test_{go[3:]}_{iter_idx}.pkl')
            true_labels = np.array(true_labels_df['go_terms'].tolist())
            true_labels = true_labels[:, i]

            try:
                predictions_df = pd.read_pickle(f'{result_path}/predictions/predictions_{go[3:]}_{iter_idx}.pkl')
                predictions = np.array(predictions_df['predictions'].tolist())
                binary_predictions = (predictions >= threshold).astype(int)

                try:
                    precision = precision_score(true_labels, binary_predictions, zero_division=0)
                    recall = recall_score(true_labels, binary_predictions, zero_division=0)
                    f1 = f1_score(true_labels, binary_predictions, zero_division=0)
                    f.write(f"{go},{precision:.4f},{recall:.4f},{f1:.4f}\n")

                except ValueError:
                    f.write(f"{go},NaN,NaN,NaN\n")

            except:
                continue


def get_counts(path, root_go, reduced_term_file, term_file, output_file, iter_idx):
    df = pd.read_csv(reduced_term_file)
    go_terms = df['Term'].tolist()

    with open(output_file, 'w') as f:
        f.write("GO_Term,Ones,Zeros\n")

        for go in go_terms:
            if go == root_go:
                continue

            i = nn.get_index_by_term(term_file, go)

            try:
                df = pd.read_pickle(f'{path}/train/train_{go[3:]}_{iter_idx}.pkl')
                gt = np.array(df['go_terms'].tolist())

                ones = np.sum(gt == 1, axis=0)
                zeros = np.sum(gt == 0, axis=0)

                f.write(f"{go},{ones[i]},{zeros[i]}\n")
                print(f"GO term: {go} -> 1s: {ones[i]}\t0s: {zeros[i]}")

            except:
                print(f'{go} train not found')


def get_balance(row):
    ones = row['Ones']
    zeros = row['Zeros']

    ratio = ones / (ones + zeros)
    if ratio <= 0.5:
        return 2 * ratio
    else:
        return 2 * (1 - ratio)


def get_results_table(eval_file, neg_eval_file, f1_file, counts_file, times_file, output_file, data, lr, freeze):
    eval_df = pd.read_csv(eval_file, dtype={'GO_Term': str})
    neg_eval_df = pd.read_csv(neg_eval_file, dtype={'GO_Term': str})
    counts_df = pd.read_csv(counts_file, dtype={'GO_Term': str})
    times_df = pd.read_csv(times_file, dtype={'GO_Term': str})
    f1_df = pd.read_csv(f1_file, dtype={'GO_Term': str})

    eval_df['AUROC_difference'] = eval_df['AUROC'] - neg_eval_df['AUROC']

    eval_df = eval_df.drop_duplicates(subset='GO_Term', keep='first')
    counts_df = counts_df.drop_duplicates(subset='GO_Term', keep='first')

    merged_df = pd.merge(eval_df, f1_df, on='GO_Term', how='left')

    merged_df = pd.merge(merged_df, counts_df, on='GO_Term', how='inner')

    merged_df['Size'] = merged_df['Ones'] + merged_df['Zeros']

    merged_df['TL'] = int(nn.tl)
    merged_df['Balance'] = merged_df.apply(get_balance, axis=1)
    merged_df['Balance'] = merged_df['Balance'].round(4)
    merged_df['Ontology'] = nn.dom
    merged_df['Frozen'] = freeze
    merged_df['LR'] = lr
    merged_df['Data'] = data
    merged_df['AUROC_difference'] = merged_df['AUROC_difference'].round(4)
    merged_df = pd.merge(merged_df, times_df, left_on='GO_Term', right_on='Term', how='inner')
    merged_df.drop(columns='Term', inplace=True)
    merged_df['Time'] = merged_df['Time'].round(4)

    if not nn.tl:
        merged_df['AUROC_difference'] = 'NaN'

    if os.path.exists(output_file):
        merged_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        merged_df.to_csv(output_file, mode='w', header=True, index=False)


def results_into_one(directory, output_csv):
    header_written = False
    with open(output_csv, "w") as outfile:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.endswith(".csv"):
                with open(file_path, "r") as infile:
                    header = infile.readline()
                    if not header_written:
                        outfile.write(header)
                        header_written = True
                    for line in infile:
                        outfile.write(line)


def combine_results(files, output_csv):
    header_written = False
    with open(output_csv, "w") as outfile:
        for file in files:
            if os.path.isfile(file) and file.endswith(".csv"):
                with open(file, "r") as infile:
                    header = infile.readline()
                    if not header_written:
                        outfile.write(header)
                        header_written = True
                    for line in infile:
                        outfile.write(line)
