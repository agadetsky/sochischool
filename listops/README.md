This is the code for our Unsupervised Parsing on Listops experiment.
The repository benefitted from various publicly available repositories
including:
- https://github.com/andre-martins/TurboParser
- https://github.com/facebookresearch/latent-treelstm

The sub-directory chuliu_edmonds contains a C implementation for the
Chuliu-Edmonds algorithm.

The sub-directory kruskals contains a C implementation for Kruskal's algorithm.

The sub-directory data_processing contains code to generate the modified
ListOps dataset (run make_all_data.py)

The sub-directory model_modules contains all implementations and models we
used in our experiments.

The scripts train.py and evaluate.py contain code to train and evaluate models. 
