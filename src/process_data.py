import pandas as pd
import tensorflow as tf
import obonet

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict, deque

MAX_SEQ_LEN = 2000
go_graph = obonet.read_obo('resources/go.obo', ignore_obsolete=False)


# Fasta file of protein_ids and sequences -> dictionary
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


def get_go_terms_indices(go_terms):
    go_indices = {}
    for i, go_term in enumerate(go_terms):
        go_indices[go_term] = i
    return go_indices


def create_annotations_dict(mapping_file, go_terms):
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
                protein_annotations = ""
            if go_term in go_terms:
                if protein_annotations != "":
                    protein_annotations += ", " + go_term
                else:
                    protein_annotations += go_term
            annotations[protein] = protein_annotations
    if current_protein is not None:
        annotations[current_protein] = protein_annotations
    return annotations


# Fasta file of protein_ids and sequences -> pickled file
def fasta_to_df(file, mapping_file, out_file, go_terms):
    prot_seq_dict = parse_fasta_to_dict(file)
    go_indices = get_go_terms_indices(go_terms)
    annotations = create_annotations_dict(mapping_file, go_indices)
    df_temp = pd.DataFrame(list(prot_seq_dict.items()), columns=['protein_id', 'sequence'])
    df_temp['go_terms'] = df_temp['protein_id'].map(annotations)
    df = df_temp[(df_temp['go_terms'].str.len() > 0)]
    df.to_pickle(out_file)


def process_dataset(df_in, df_out, sorted_terms):
    df = pd.read_pickle(df_in)

    sequences = df['sequence'].tolist()
    go_terms = df['go_terms'].tolist()

    # Tokenize the sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sequences)
    sequences_numeric = tokenizer.texts_to_sequences(sequences)

    # Pad sequences to have the same length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences_numeric, maxlen=MAX_SEQ_LEN,
                                                                     padding='post')
    # Convert GO terms to binary representation
    mlb = MultiLabelBinarizer()
    go_terms_binary = mlb.fit_transform([str(terms).split(', ') for terms in go_terms])

    curr_order = list(mlb.classes_)
    reorder_indices = [curr_order.index(term) for term in sorted_terms]
    reordered_go_terms_binary = go_terms_binary[:, reorder_indices]

    df_processed = pd.DataFrame({'sequence': padded_sequences.tolist(),
                                 'go_terms': reordered_go_terms_binary.tolist()})

    df_processed.to_pickle(df_out)


def load_mapping(file_path):
    mapping = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                mapping.append((parts[0], parts[1], parts[2]))
    return mapping


def save_mapping(mapping, output_file):
    with open(output_file, 'w') as file:
        for entry in mapping:
            file.write('\t'.join(entry) + '\n')


def get_ancestors(go_term, visited=None):
    if visited is None:
        visited = set()

    ancestors = []
    if go_term not in visited:
        visited.add(go_term)
        for child, parent, key in go_graph.out_edges(go_term, keys=True):
            if key == 'is_a':
                ancestors.append(parent)
                ancestors.extend(get_ancestors(parent, visited))

    return ancestors


def build_alt_id_mapping():
    alt_id_mapping = {}
    for node, data in go_graph.nodes(data=True):
        alt_ids = data.get('alt_id', [])
        for alt_id in alt_ids:
            alt_id_mapping[alt_id] = node
    return alt_id_mapping


def build_obsolete_mapping():
    obsolete_mapping = {}
    for node, data in go_graph.nodes(data=True):
        if data.get('is_obsolete', False):
            replaced_by = data.get('replaced_by')
            if replaced_by:
                obsolete_mapping[node] = replaced_by[0]
            else:
                obsolete_mapping[node] = 'X'
    return obsolete_mapping


def replace_terms(mapping, replacement_mapping):
    updated_mapping = []
    for protein, go_term, category in mapping:
        primary_id = replacement_mapping.get(go_term, go_term)
        if primary_id != 'X':
            updated_mapping.append((protein, primary_id, category))
    return updated_mapping


def update_mapping_ancestors(mapping):
    updated_mapping = []

    for protein, go_term, category in mapping:
        ancestors = get_ancestors(go_term)
        updated_mapping.append((protein, go_term, category))
        for ancestor in ancestors:
            updated_mapping.append((protein, ancestor, category))
    return updated_mapping


def remove_duplicates(mapping_file, output_file):
    lines_seen = set()
    with open(mapping_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if line not in lines_seen:
                f_out.write(line)
                lines_seen.add(line)


def sort_terms_kahn(paths, output_file):
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    parents = defaultdict(set)

    for path in paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            graph[u].append(v)
            in_degree[v] += 1
            in_degree.setdefault(u, 0)
            parents[v].add(u)
            parents.setdefault(u, set())

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []
    term_parents = {}

    while queue:
        current = queue.popleft()
        topo_order.append(current)
        term_parents[current] = ', '.join(sorted(parents[current])) if parents[current] else 'X'
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    print("Sorted terms:", topo_order)

    df = pd.DataFrame({
        'Term': topo_order,
        'Index': range(len(topo_order)),
        'Parents': [term_parents[term] for term in topo_order]
    })

    df.to_csv(output_file, index=False)

    return topo_order


# Fix the non-hierarchical mapping
def update_mappings():
    alt_id_mapping = build_alt_id_mapping()

    mapping = load_mapping('resources/CAFA3_training_data/uniprot_sprot_exp.txt')
    updated_mapping = replace_terms(mapping, alt_id_mapping)
    save_mapping(updated_mapping, 'resources/mapping_no_alt_ids.txt')

    obsolete_mapping = build_obsolete_mapping()
    mapping = load_mapping('resources/mapping_no_alt_ids.txt')
    updated_mapping = replace_terms(mapping, obsolete_mapping)
    save_mapping(updated_mapping, 'resources/mapping_updated.txt')

    mapping = load_mapping('resources/mapping_updated.txt')
    updated_mapping = update_mapping_ancestors(mapping)
    save_mapping(updated_mapping, 'resources/mapping_extended.txt')

    remove_duplicates('resources/mapping_extended.txt', 'resources/mapping.txt')

