# 🎓 Protein Function Prediction using CNN | Bachelor's Thesis

Welcome to the GitHub repository for my Bachelor’s Thesis at the Czech Technical University in Prague, Faculty of Electrical Engineering.

Title: Automated Protein Annotation with Integration of Gene Ontology Inter-Relationships

This project explores deep learning methods to predict Gene Ontology (GO) terms from protein sequences using hierarchical transfer learning.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)

---

## 📘 Overview

This thesis implements a CNN-based approach to predict GO annotations from amino acid sequences. It includes preprocessing, training across hierarchical paths, and comparing against BLAST+kNN and PRIORS baselines.

---

## 📝 Abstract

> Automated protein function prediction is essential for efficiently annotating large-scale genomic data. This thesis proposes a novel approach, which integrates a convolutional neural network with transfer learning to assign Gene Ontology (GO) terms to protein sequences. The convolutional neural network is designed to process GO terms arranged in GO graph paths individually according to their hierarchical distance from the root term. The hierarchical relationships between GO terms are leveraged to reduce the dataset at every level, thereby significantly streamlining the training process. To assess the efficacy of the proposed method, it is benchmarked against a range of existing approaches for automated protein function annotation.

---

## 🗂️ Project Structure

```bash
.
├── resources/             
├── results/               
├── src/                 
│   ├── process_data.py
│   ├── utils.py
│   ├── nn.py
│   ├── blast_knn.py
│   ├── esm_classifier.py
│   └── priors.py
├── thesis.pdf              
└── README.md               
``` 

---

## 🛠️ Requirements
```bash
Python 3.10+
numpy
pandas
tensorflow 2.10
scikit-learn
obonet
```

---

## 🚀 How to Run

1. Update mappings with
    ```bash
    process_data.update_mappings()
    ```
2. Select GO terms to train on and generate all possible paths
3. Create data.pkl with 
    ```bash
    process_data.fasta_to_df(fasta_file, 
                            updated_mapping_file, 
                            output_file=data.pkl, 
                            selected_GO_terms)
    ```
4. For comparison it's needed to first train the NN without transfer learning
    ```bash
    nn.run_nn(paths_file,
            selected_ontology, 
            transfer=False, 
            output_directory, 
            dataset_reduction_type=original, 
            learning_rate, 
            freeze_layers=False)
    ```
5. Train NN with any parameters 
    ```bash
    nn.run_nn( ... )
    ```
6. Evaluate results with functions from utils.py

Because of the term-by-term training and saving data for results, models and datasets are saved for every iteration of every term; this can lead to hundreds of GBs of space.

---

## 📊 Results




