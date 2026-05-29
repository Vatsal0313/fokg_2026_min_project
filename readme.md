# FOkg Mini Project – Fact Checking with DBpedia

## AUC SCORE gerbil: https://gerbil-kbc.aksw.org/gerbil/experiment?id=202605290001

## Overview

This project implements a fact‑checking engine that queries DBpedia via SPARQL and returns a veracity score (0–1) for RDF triples.  
The system uses:

- `parse_data.py` – load N‑Triples files and extract statements.
- `fact_checker.py` – core scoring logic (SPARQL + predicate‑level priors).
- `generate_result.py` – run the fact checker on a test file and save predictions.
- `compute_train_auc.py` – evaluate performance on the training set.


## For small dataset:
python generate_result.py --test data/small_test.nt --out output/small_result.ttl

## Re‑running the Complete Pipeline

If you need to regenerate everything from scratch, use the `--force` flag to ignore existing caches and outputs:

```bash
python generate_result_force.py --force
python compute_train_auc_plot.py --force