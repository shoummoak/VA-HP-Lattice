# Lattice Protein Folding with Variational Annealing

This is the code respository for the research paper titled `Lattice Protein Folding with Variational Annealing`. See [arxiv](https://arxiv.org/abs/2502.20632) or [Journal](https://iopscience.iop.org/article/10.1088/2632-2153/adf376).

## Code Summary
- The code is set up to train on any one of the HP protein chains listed in `main_cli.py`
- After training, the training data (sampled folds along with their energies and log-probabilities) is saved in the folder specified by the path provided when calling `main_cli.py`
- 


## Python dependencies
- tensorflow v2.15.1
- tf_agents v0.19.0

## How to Run Code
### Option 1: Run from command line
- main_cli.py and train.py should be in the same folder (unless filepaths are changed within code)
- To train, for example, on the 20merA sequence with N_anneal=10,000, N_warmup=1000, T0=1.0, seed=111, save the training data in folderA, trained model in folderB, we run the command below in the command line interface:
- `python main_cli.py --protein-key 20merA --n-warmup 1000 --n-anneal 10000 --n-train 5 --path-data-folder /.../folderA/ --path-model /.../folderB --T0 1 --seed 111`

### Option 2: Run Python Notebook
- Pass the desired parameters where the vca object is being created and simply run the file.
