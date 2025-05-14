import os
import random

import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import numpy as np

import process_data

keras = tf.keras
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Concatenate, Input, \
    Activation, GlobalAvgPool1D, Embedding
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.models import Model
from sklearn.model_selection import KFold, train_test_split


MAX_SEQ_LEN = 2000
AA_LEN = 26

tl = False
directory = ''
dom = ''
data_reduction = ''
lr = 1e-4


def create_model(freeze):
    filters = [256, 128]
    kernel_size = 8

    input = Input(shape=(MAX_SEQ_LEN,), dtype=np.float32)
    if freeze:
        net = Embedding(input_dim=AA_LEN, output_dim=AA_LEN, input_length=MAX_SEQ_LEN, trainable=False)(input)  # False
    else:
        net = Embedding(input_dim=AA_LEN, output_dim=AA_LEN, input_length=MAX_SEQ_LEN, trainable=True)(input)

    if freeze:
        net = Conv1D(filters=filters[0], kernel_size=kernel_size, activation='relu', trainable=False)(net)
        net = Dropout(0.3)(net)
        net = MaxPooling1D()(net)

        net = Conv1D(filters=filters[1], kernel_size=kernel_size, activation='relu', trainable=True)(net)
        net = Dropout(0.3)(net)
        net = MaxPooling1D()(net)

    else:
        for f in filters:
            net = Conv1D(filters=f, kernel_size=kernel_size, activation='relu', trainable=True)(net)
            net = Dropout(0.3)(net)
            net = MaxPooling1D()(net)

    net = Dropout(0.5)(net)
    net = GlobalAvgPool1D()(net)
    # net = Flatten()(net)
    # net = Dropout(0.5)(net)
    net = Dense(units=1)(net)
    net = BatchNormalization()(net)
    output = Activation(activation='sigmoid')(net)
    # output = Dense(units=1, activation='sigmoid')(net)

    model = Model(inputs=input, outputs=output)

    return model


def random_substitute(sequence, substitution_rate=0.05):
    augmented_seq = sequence.copy()
    num_substitutions = int(len(sequence) * substitution_rate)
    for _ in range(num_substitutions):
        index = random.randint(0, len(sequence) - 1)
        augmented_seq[index] = random.randint(1, 20)
    return augmented_seq


def local_shuffle(sequence, window_size=5):
    augmented_seq = sequence.copy()
    for i in range(0, len(sequence), window_size):
        window = augmented_seq[i:i + window_size]
        np.random.shuffle(window)
        augmented_seq[i:i + window_size] = window
    return augmented_seq


def random_insertion(sequence, insertion_rate=0.02):
    augmented_seq = sequence.copy()

    num_insertions = int(len(sequence) * insertion_rate)
    for _ in range(num_insertions):
        index = random.randint(0, len(augmented_seq) - 1)
        np.insert(augmented_seq, index, random.randint(1, 20))

    return np.array(augmented_seq)


def random_deletion(sequence, deletion_rate=0.02):
    augmented_seq = sequence.copy()

    num_deletions = int(len(sequence) * deletion_rate)
    for _ in range(num_deletions):
        if len(augmented_seq) > 1:
            index = random.randint(0, len(augmented_seq) - 1)
            np.delete(augmented_seq, index)

    return np.array(augmented_seq)


def augment_data(df, len_needed):
    df_to_augment = df.copy()

    rows_needed = len_needed - len(df_to_augment)
    rows_to_augment = df_to_augment.sample(n=rows_needed, replace=True)
    augmented_rows = []

    for _, row in rows_to_augment.iterrows():
        augmentation_function = random.choice([random_substitute,
                                               local_shuffle,
                                               random_insertion,
                                               random_deletion])
        new_sequence = augmentation_function(row['sequence'])
        augmented_rows.append({'sequence': new_sequence, 'go_terms': row['go_terms']})

    df_to_augment = pd.concat([df_to_augment, pd.DataFrame(augmented_rows)], ignore_index=True)

    return df_to_augment


def get_parents_by_term(term_file, term):
    df = pd.read_csv(term_file)
    row = df[df['Term'] == term]

    if not row.empty:
        parents_str = row['Parents'].values[0]
        parents = parents_str.split(', ') if parents_str != 'X' else None
        return parents
    else:
        return None


def get_index_by_term(term_file, term):
    df = pd.read_csv(term_file)
    row = df[df['Term'] == term]

    if not row.empty:
        return row.iloc[0]['Index']
    else:
        return None


# Reduce the current dataset, optionally truncating or augmenting the result
def reduce_data(train_in, train_out, val_in, val_out, parent_i, curr_i):
    parent_train_df = pd.read_pickle(train_in)
    parent_val_df = pd.read_pickle(val_in)

    train_df = pd.concat([parent_train_df, parent_val_df], ignore_index=True)

    if data_reduction == 'original':
        combined_df = train_df[train_df['go_terms'].apply(lambda x: x[parent_i] == 1)]

    elif data_reduction == 'augmented':
        filtered_df_pos = train_df[train_df['go_terms'].apply(lambda x: x[parent_i] == 1 and x[curr_i] == 1)]
        filtered_df_neg = train_df[train_df['go_terms'].apply(lambda x: x[parent_i] == 1 and x[curr_i] == 0)]

        if filtered_df_neg.empty or filtered_df_pos.empty:
            combined_df = train_df
        elif len(filtered_df_pos) > len(filtered_df_neg):
            augmented_df = augment_data(filtered_df_neg, len(filtered_df_pos))
            combined_df = pd.concat([augmented_df, filtered_df_pos], ignore_index=True)
        else:
            augmented_df = augment_data(filtered_df_pos, len(filtered_df_neg))
            combined_df = pd.concat([augmented_df, filtered_df_neg], ignore_index=True)

    elif data_reduction == 'truncated':   # if data_reduction == truncated
        filtered_df_pos = train_df[train_df['go_terms'].apply(lambda x: x[parent_i] == 1 and x[curr_i] == 1)]
        filtered_df_neg = train_df[train_df['go_terms'].apply(lambda x: x[parent_i] == 1 and x[curr_i] == 0)]

        if filtered_df_neg.empty or filtered_df_pos.empty:
            combined_df = train_df
        elif len(filtered_df_pos) > len(filtered_df_neg):
            truncated_df = filtered_df_pos.sample(n=min(len(filtered_df_neg), len(filtered_df_pos)))
            combined_df = pd.concat([truncated_df, filtered_df_neg], ignore_index=True)
        else:
            truncated_df = filtered_df_neg.sample(n=min(len(filtered_df_pos), len(filtered_df_neg)))
            combined_df = pd.concat([truncated_df, filtered_df_pos], ignore_index=True)

    sequences = combined_df['sequence'].tolist()
    go_terms = combined_df['go_terms'].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        sequences, go_terms, test_size=0.1, random_state=42
    )

    train_df = pd.DataFrame({'sequence': X_train, 'go_terms': y_train})
    val_df = pd.DataFrame({'sequence': X_val, 'go_terms': y_val})

    train_df.to_pickle(train_out)
    val_df.to_pickle(val_out)


def choose_parent_random(parents):
    return parents[random.randint(0, len(parents) - 1)]


def cnn(train, val, test, curr_go, parent_go, term_file, iter_idx, freeze):
    model_path = f'results/{dom}/{directory}/models/model_{curr_go[3:]}_{iter_idx}.h5'

    if os.path.exists(model_path):
        return

    train_df = pd.read_pickle(train)
    val_df = pd.read_pickle(val)
    test_df = pd.read_pickle(test)

    X_train = np.array(train_df['sequence'].tolist())
    y_train_temp = np.array(train_df['go_terms'].tolist())
    i = get_index_by_term(term_file, curr_go)
    y_train = y_train_temp[:, i][:, np.newaxis]

    X_val = np.array(val_df['sequence'].tolist())
    y_val_temp = np.array(val_df['go_terms'].tolist())
    y_val = y_val_temp[:, i][:, np.newaxis]

    X_test = np.array(test_df['sequence'].tolist())
    y_test_temp = np.array(test_df['go_terms'].tolist())
    y_test = y_test_temp[:, i][:, np.newaxis]

    epochs = 50
    batch_size = 32

    print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
        f'\nEpoch {epoch + 1}/{epochs}, Loss: {logs["loss"]}, Accuracy: {logs["accuracy"]}'))
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True)

    model = create_model(freeze)

    if tl:
        try:
            model.load_weights(f'results/{dom}/{directory}/models/model_{parent_go[3:]}_{iter_idx}.h5')
        except:
            model.load_weights(f'results/{dom}/no_tl/models/model_{parent_go[3:]}_{iter_idx}.h5')

    loss = 'binary_crossentropy'
    optimizer = Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print(model.summary())

    print('>>> training started <<<')
    print(dt.now().time())

    start_time = dt.now()
    model.fit(X_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[print_callback, early_stopping_callback])
    end_time = dt.now()
    training_time = end_time - start_time
    print('>>> training finished <<<')
    print(dt.now().time())

    if not os.path.exists(f'results/{dom}/{directory}/models/'):
        os.makedirs(f'results/{dom}/{directory}/models/')
    model.save(f'results/{dom}/{directory}/models/model_{curr_go[3:]}_{iter_idx}.h5')
    if tl:
        model.save_weights(f'results/{dom}/{directory}/models/modelweights_{curr_go[3:]}_{iter_idx}.h5')
    print('model saved')

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    if not os.path.exists(f'results/{dom}/{directory}/predictions/'):
        os.makedirs(f'results/{dom}/{directory}/predictions/')
    if not os.path.exists(f'results/{dom}/{directory}/times/'):
        os.makedirs(f'results/{dom}/{directory}/times/')

    predictions = model.predict(X_test)
    df = pd.DataFrame({'predictions': predictions.flatten()})
    df.to_pickle(f'results/{dom}/{directory}/predictions/predictions_{curr_go[3:]}_{iter_idx}.pkl')

    with open(f'results/{dom}/{directory}/times/trainingtime_{curr_go[3:]}_{iter_idx}.txt', 'w') as f:
        f.write(str(training_time))


# Run CNN on selected paths, ontology, with or without TL
#   data_reduct = type of dataset reduction; original, augmented, truncated
#   learn_rate = chosen learning rate
#   freeze_layers = if True freezes the embedding and the first convolutional layer of CNN
def run_nn(paths_file, ontology, transfer, out_directory, data_reduct, learn_rate, freeze_layers):
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     print(f'CUDA is available. Number of GPUs: {len(physical_devices)}')
    #     for gpu in physical_devices:
    #         print(f' - GPU: {gpu}')
    # else:
    #     print('CUDA is not available. No GPU detected.')

    # Set the global variables with values
    global dom, tl, directory, data_reduction, lr
    dom = ontology
    tl = transfer
    directory = out_directory
    data_reduction = data_reduct
    lr = learn_rate

    root_go = 'GO:0003674' if dom == 'mf' else 'GO:0008150' if dom == 'bp' else 'GO:0005575'
    freeze = freeze_layers

    with open(paths_file, 'r') as file:
        paths = []
        for line in file:
            if line.startswith('#'):
                continue
            p = line.strip().split(', ')
            paths.append(p)

    all_go_terms_sorted = []

    # Obtain topologically sorted terms from paths and save them into a file
    if tl:
        paths_arr = [paths]

        current = paths
        while any(current):
            current = [sub_path[1:] for sub_path in current if len(sub_path[1:]) >= 2]
            if current:
                paths_arr.append(current)

        for level_idx in range(len(paths_arr)):
            if not os.path.exists(f'resources/{dom}/{directory}/from_{level_idx}/'):
                os.makedirs(f'resources/{dom}/{directory}/from_{level_idx}/')
            print(f"{level_idx} = {paths_arr[level_idx]}")
            go_terms_sorted = process_data.sort_terms_kahn(paths_arr[level_idx],
                                                           f'resources/{dom}/{directory}/from_{level_idx}/sorted_terms_{dom}.csv')
            all_go_terms_sorted.append(go_terms_sorted)

    else:
        if not os.path.exists(f'resources/{dom}/{directory}'):
            os.makedirs(f'resources/{dom}/{directory}')
        go_terms_sorted = process_data.sort_terms_kahn(paths,
                                                       f'resources/{dom}/{directory}/sorted_terms_{dom}.csv')
        all_go_terms_sorted.append(go_terms_sorted)

    process_data.fasta_to_df('resources/CAFA3_training_data/uniprot_sprot_exp.fasta',
                             'resources/mapping.txt',
                             f'resources/{dom}/data.pkl',
                             all_go_terms_sorted[0])

    n_splits = 5
    validation_ratio = 0.1

    # Run with TL
    if tl:
        old_directory = directory

        for go_terms_sorted in all_go_terms_sorted:
            paths_idx = all_go_terms_sorted.index(go_terms_sorted)

            if paths_idx == 0:
                continue

            directory = f'{old_directory}/from_{paths_idx}'

            if not os.path.exists(f'resources/{dom}/{directory}/train/'):
                os.makedirs(f'resources/{dom}/{directory}/train/')
            if not os.path.exists(f'resources/{dom}/{directory}/val/'):
                os.makedirs(f'resources/{dom}/{directory}/val/')

            # For every term
            for i in range(0, len(go_terms_sorted)):
                curr_go = go_terms_sorted[i]

                # Use root term to create the dataset for the ontology (dom)
                if curr_go == root_go:
                    process_data.process_dataset(f'resources/{dom}/data.pkl',
                                                 f'resources/{dom}/{directory}/data.pkl',
                                                 go_terms_sorted)
                    continue
                else:
                    parents = get_parents_by_term(f'resources/{dom}/{directory}/sorted_terms_{dom}.csv',
                                                     curr_go)
                    if not parents:
                        continue

                    # For n splits (from n-fold cross validation on training without TL)
                    for j in range(1, n_splits + 1):
                        curr_idx = get_index_by_term(f'resources/{dom}/{directory}/sorted_terms_{dom}.csv',
                                                        curr_go)
                        parent_go = parents[0] if len(parents) == 1 else choose_parent_random(parents)
                        parent_idx_all = get_index_by_term(
                            f'resources/{dom}/{old_directory}/from_0/sorted_terms_{dom}.csv',
                            parent_go)
                        curr_idx_all = get_index_by_term(
                            f'resources/{dom}/{old_directory}/from_0/sorted_terms_{dom}.csv',
                            curr_go)

                        if parent_go != root_go:
                            if os.path.exists(f'resources/{dom}/{directory}/train/train_{parent_go[3:]}_{j}.pkl'):
                                reduce_data(f'resources/{dom}/{directory}/train/train_{parent_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/{directory}/train/train_{curr_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/{directory}/val/val_{parent_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/{directory}/val/val_{curr_go[3:]}_{j}.pkl',
                                               parent_idx_all,
                                               curr_idx_all)
                            else:
                                reduce_data(f'resources/{dom}/no_tl/train/train_{parent_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/{directory}/train/train_{curr_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/no_tl/val/val_{parent_go[3:]}_{j}.pkl',
                                               f'resources/{dom}/{directory}/val/val_{curr_go[3:]}_{j}.pkl',
                                               parent_idx_all,
                                               curr_idx_all)
                        else:
                            continue

                        train = f'resources/{dom}/{directory}/train/train_{curr_go[3:]}_{j}.pkl'
                        val = f'resources/{dom}/{directory}/val/val_{curr_go[3:]}_{j}.pkl'
                        test = f'resources/{dom}/no_tl/test/test_{curr_go[3:]}_{j}.pkl'

                        cnn(train,
                               val,
                               test,
                               curr_go,
                               parent_go,
                               f'resources/{dom}/{old_directory}/from_0/sorted_terms_{dom}.csv',
                               j,
                               freeze)

    # Run without TL
    else:
        go_terms_sorted = all_go_terms_sorted[0]

        for i in range(1, len(go_terms_sorted)):
            curr_go = go_terms_sorted[i]

            kf = KFold(n_splits=n_splits, shuffle=False)

            process_data.process_dataset(f'resources/{dom}/data.pkl',
                                         f'resources/{dom}/{directory}/data.pkl',
                                         go_terms_sorted)

            df_data = pd.read_pickle(f'resources/{dom}/{directory}/data.pkl')

            sequences = df_data['sequence'].tolist()
            go_terms = df_data['go_terms'].tolist()

            sequences = np.array(sequences)
            go_terms = np.array(go_terms)

            fold_idx = 1
            for train_index, test_index in kf.split(sequences):
                # Split data into training and testing for this fold
                X_train_full, X_test = sequences[train_index], sequences[test_index]
                y_train_full, y_test = go_terms[train_index], go_terms[test_index]

                # Further split training data into training and validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, test_size=validation_ratio, random_state=42
                )

                train_df = pd.DataFrame({'sequence': X_train.tolist(), 'go_terms': y_train.tolist()})
                val_df = pd.DataFrame({'sequence': X_val.tolist(), 'go_terms': y_val.tolist()})
                test_df = pd.DataFrame({'sequence': X_test.tolist(), 'go_terms': y_test.tolist()})

                if not os.path.exists(f'resources/{dom}/{directory}/train/'):
                    os.makedirs(f'resources/{dom}/{directory}/train/')
                if not os.path.exists(f'resources/{dom}/{directory}/val/'):
                    os.makedirs(f'resources/{dom}/{directory}/val/')
                if not os.path.exists(f'resources/{dom}/{directory}/test/'):
                    os.makedirs(f'resources/{dom}/{directory}/test/')

                train_df.to_pickle(f'resources/{dom}/{directory}/train/train_{curr_go[3:]}_{fold_idx}.pkl')
                val_df.to_pickle(f'resources/{dom}/{directory}/val/val_{curr_go[3:]}_{fold_idx}.pkl')
                test_df.to_pickle(f'resources/{dom}/{directory}/test/test_{curr_go[3:]}_{fold_idx}.pkl')

                cnn(f'resources/{dom}/{directory}/train/train_{curr_go[3:]}_{fold_idx}.pkl',
                       f'resources/{dom}/{directory}/val/val_{curr_go[3:]}_{fold_idx}.pkl',
                       f'resources/{dom}/{directory}/test/test_{curr_go[3:]}_{fold_idx}.pkl',
                       curr_go,
                       parent_go=None,
                       term_file=f'resources/{dom}/{directory}/sorted_terms_{dom}.csv',
                       iter_idx=fold_idx,
                       freeze=freeze)

                fold_idx += 1
