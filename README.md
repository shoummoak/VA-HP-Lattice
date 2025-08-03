# VA-HP-Lattice

## Research Article
This is the code respository for the research article **"Lattice Protein Folding with Variational Annealing"**. [arxiv](https://arxiv.org/abs/2502.20632)

## Python dependencies
Please see dependencies.txt

## How to Run Code
**Option 1:** Run from command line
- main_cli.py and train.py should be in the same folder (unless filepaths are changed within code)
- run main_cli.py with arguments like the example below

`--protein-key 20merA --n-warmup 1000 --n-anneal 10000 --n-train 5 --path-data-folder /home/data/ --path-model /home/data/ --T0 1 --seed 111`
