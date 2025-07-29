
# imports
import argparse
import os
from train import *

# cli argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration")
    # HP chain
    parser.add_argument('--protein-key', type=str, default='18merA',
                       help='Protein chain identifier')
    # VA hyperparameters
    parser.add_argument('--n-warmup', type=int, default=20,
                       help='Number of warmup steps')
    parser.add_argument('--n-anneal', type=int, default=20,
                       help='Number of annealing steps')
    parser.add_argument('--n-train', type=int, default=5,
                       help='Number of training steps')
    parser.add_argument('--T0', type=float, default=10.0,
                       help='Initial temperature')
    parser.add_argument('--annealer', type=str, default='linear', 
                        help='annealing function between linear and inverse')
    parser.add_argument('--inverse_exponent', type=float, default=1.0, 
                        help='exponent of the inverse annealer; not required for linear annealer')
    # paths
    parser.add_argument('--path-model', type=str, default='./trained_model/model.ckpt',
                       help='Path to save/load model checkpoint')
    parser.add_argument('--path-data-folder', type=str, default='./data',
                       help='Path to data folder')
    # seed
    parser.add_argument('--seed', type=int, default=111,
                       help='seed for reproducibility')
    return parser.parse_args()


""" Main """
if __name__ == "__main__":

    args = parse_args()
    # HP benchmarks from Istrail Lab: https://www.brown.edu/Research/Istrail_Lab/hp2dbenchmarks.html
    # now unavailable
    Istrail_chains = {
        '18merA': 'HHPPPPPHHPPPHPPPHP',
        '18merB': 'HPHPHHHPPPHHHHPPHH',
        '18merC': 'PHPPHPHHHPHHPHHHHH',
        '20merA': 'HPHPPHHPHPPHPHHPPHPH',
        '20merB': 'HHHPPHPHPHPPHPHPHPPH',
        '24mer': 'HHPPHPPHPPHPPHPPHPPHPPHH',
        '25mer': 'PPHPPHHPPPPHHPPPPHHPPPPHH',
        '36mer': 'PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP',
        '48mer': 'PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH',
        '50mer': 'HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH',
        '60mer': 'PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP',
        '64mer': 'HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH',
        '85mer': 'HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH',
        '100merA': 'PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH',
        '100merB': 'PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH'
    }
    protein_key = args.protein_key
    protein = [2 if bead == 'P' else 1 for bead in Istrail_chains[protein_key]]     # encode chains: 1:H, 2:P
    N = len(protein)
    model = vca(N=N, protein=protein, n_warmup=args.n_warmup, n_anneal=args.n_anneal, n_train=args.n_train, 
                T0=args.T0, annealer=args.annealer, inv_exp=args.inverse_exponent, ckpt_path=args.path_model, seed=args.seed)
    energies_train, samples_train, logprobs_train, temperatures = model.test_run()

    # post-processing
    # os.makedirs(args.path_data_folder, exist_ok=True)
    np.save(os.path.join(args.path_data_folder, "energies_training.npy"), energies_train)
    np.save(os.path.join(args.path_data_folder, "logprobs_training.npy"), logprobs_train)
    np.save(os.path.join(args.path_data_folder, "folds_training.npy"), get_unique_gs_folds(samples_train, energies_train))
    # np.save(os.path.join(args.path_data_folder, "energies_cooled.npy"), energies_cooled)
    # np.save(os.path.join(args.path_data_folder, "logprobs_cooled.npy"), logprobs_cooled)
    # np.save(os.path.join(args.path_data_folder, "folds_cooled.npy"), get_unique_gs_folds(samples_cooled, energies_cooled))
    np.save(os.path.join(args.path_data_folder, "anneal_schedule.npy"), temperatures)

