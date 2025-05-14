# Utility functions to create and evaluate PRIORs algorithm for comparison.


import pandas as pd


def read_mapping_file_and_save_df(file_path, selected_go_terms, output_file):
    mapping_data = pd.read_csv(file_path, sep="\t", header=None, names=["ProteinID", "GO_Term", "Category"])
    mapping_data = mapping_data.drop(columns=["Category"])
    unique_proteins = mapping_data["ProteinID"].unique()
    binary_matrix = pd.DataFrame(0, index=unique_proteins, columns=selected_go_terms)

    for _, row in mapping_data.iterrows():
        if row["GO_Term"] in selected_go_terms:
            binary_matrix.loc[row["ProteinID"], row["GO_Term"]] = 1

    if output_file.endswith(".pkl"):
        binary_matrix.to_pickle(output_file)
    else:
        binary_matrix.to_csv(output_file)
    print("")
    return binary_matrix


def calculate_priors_and_save(input_file, output_file):
    if input_file.endswith(".pkl"):
        binary_matrix = pd.read_pickle(input_file)
    else:
        binary_matrix = pd.read_csv(input_file, index_col=0)

    go_term_counts = binary_matrix.sum(axis=0)
    total_proteins = len(binary_matrix)
    go_term_priors = go_term_counts / total_proteins

    predictions = pd.DataFrame(index=binary_matrix.index, columns=binary_matrix.columns)
    for go_term in binary_matrix.columns:
        predictions[go_term] = go_term_priors[go_term]

    if output_file.endswith(".pkl"):
        predictions.to_pickle(output_file)
    else:
        predictions.to_csv(output_file)

    return predictions