# imports
import numpy as np
import tensorflow as tf     # version 2.15.1
import tf_agents            # version 0.19.0
import random
import time

"""
DilatedRNN Class
"""
class DilatedRNN(object):
    def __init__(self, systemsize, cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation=tf.nn.relu, units=[2], scope='DilatedRNN', seed=112):
        """
            systemsize:  length of protein - 1
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            activation:  activation of the RNN cell
            seed:        pseudo-random number generator
        """

        self.graph=tf.Graph()
        self.scope=scope    #Label of the RNN wavefunction
        self.N=systemsize   #Number of sites of the 1D chain
        self.numlayers = len(units)
        self.numunits = units[0]
        self.outputdim = 4
        # inputdim: one-hot vector of previous move (size 4) + one-hot vector of current bead (size 2)
        self.inputdim = 4+2
        random.seed(seed)
        np.random.seed(seed)

        # defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)

                # 
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(i)+str(n),dtype=tf.float64) for i in range(self.numlayers)] for n in range(self.N-1)]
                self.dense = [tf.compat.v1.layers.Dense(self.outputdim,activation=tf.nn.log_softmax,name='DRNN_dense'+str(n),dtype=tf.float64) for n in range(self.N-1)] #Define the Fully-Connected layer followed by a Softmax


    def projection_static(self, lprobs, mask):        # lprobs: logarithm of probabilities
        # change probability to 0 for invalid moves (equivalently to setting masked logprobs to <-200) | (numsamples, 4)
        lprobs = lprobs*mask + (-mask*1 + 1)*-35
        lprobs = lprobs - tf.reshape(tf.reduce_logsumexp(lprobs, 1), [-1,1])    # L1 normalization in log probs space; not necessary
        return lprobs     # (numsamples, 4)

    def projection_dynamic(self, lprobs, mask):        # lprobs: logarithm of probabilities | stop gradient
        lprobs = lprobs*mask + (-mask*1 + 1)*(tf.reshape(tf.math.reduce_min(lprobs, axis=1), shape=[-1, 1])-30)
        # lprobs = lprobs*mask + (-mask*1 + 1)*(tf.reshape(tf.math.reduce_min(lprobs, axis=1), shape=[-1, 1])-35)
        lprobs = lprobs - tf.reshape(tf.reduce_logsumexp(lprobs, 1), [-1,1])
        return lprobs     # (numsamples, 4)

    def projection_identity(self, lprobs, mask): # return max(min_prob, log(epsilon)
        return tf.maximum(lprobs, -20)     # (numsamples, 4)

    # !could potentially use reshaping trick here instead of loop
    def checker_func(self, previous_positions, newposition):
        mask = tf.ones(self.numsamples, dtype = tf.float64)
        for index, pos in enumerate(previous_positions):
            mask = mask * tf.cast(1.-tf.where(tf.math.equal(pos, newposition),1.,0.), tf.float64)
        # print("Done")
        return mask

    def sample(self, numsamples, protein):
        """
            generate samples from probability distribution parametrized by the RNN
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension
            ------------------------------------------------------------------------
            Returns:
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                inputs = tf.zeros((numsamples, self.inputdim), dtype=tf.float64) #Feed the table b in tf.
                # initial input to feed to the rnn
                self.numsamples = inputs.shape[0]
                rnn_states = []
                for i in range(self.numlayers):
                    for n in range(self.N):
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        # rnn_states.append(self.rnn[0][i].zero_state(self.numsamples, dtype=tf.float64)) #Initialize the RNN hidden state
                        zero_state = np.zeros((self.numsamples, self.numunits), dtype=np.float64)
                        protein_onehot = np.zeros(2, dtype=np.float64)
                        protein_onehot[protein[0]-1] = 1
                        zero_state[:, :2] = protein_onehot
                        zero_state = tf.convert_to_tensor(zero_state)
                        rnn_states.append(zero_state) #Initialize the RNN hidden state
                # zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)
                samples = []
                log_probs = []
                previous_positions = []
                # the effect of an L, R, U, D move on a position
                position = tf.constant([[2*self.N**2 + self.N] for _ in range(self.numsamples)], tf.float64)    # (numsamples, 1)
                position = tf.reshape(position, shape=[self.numsamples])       # (numsamples,)
                previous_positions.append(position)     # append default starting position
                offset = np.array([-1, 1, -2*self.N, 2*self.N], np.float64)
                mask = tf.constant([1., 1., 1., 1.], dtype = tf.float64)
                # sampling
                for n in range(1, self.N):
                    rnn_output = inputs
                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[n-1][i](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[n-1][i](rnn_output, rnn_states[i])

                    log_probs_nth = self.dense[n-1](rnn_output)     # obtain the log of probability distribution of the 4 moves for the nth move; (numsamples, 4)
                    log_probs_nth = self.projection_identity(log_probs_nth, mask)     # masking; (numsamples, 4)
                    log_probs.append(log_probs_nth)

                    # sample move from probability distribution
                    # move = tf.random.categorical(log_probs_nth, num_samples=1)    # (numsamples, 1)
                    # samples.append(tf.reshape(move, shape = [self.numsamples]))     # (numsamples,) appended
                    sampler = tf_agents.distributions.masked.MaskedCategorical(log_probs_nth, mask)     # (numsamples,)
                    move = sampler.sample()
                    samples.append(move)     # (numsamples,) appended
                    one_hot_move = tf.reshape(tf.one_hot(move, depth=4, dtype=tf.float64), shape = [self.numsamples, -1])   # (numsmaples, 1, 4) reduced to (numsamples, 4)
                    position = position + tf.reduce_sum(one_hot_move*offset, axis=1)    # (numsamples, 4) reduced to (numsamples,)
                    previous_positions.append(position)     # tensor obj with shape (numsamples,) appended
                    
                    # build RNN input for next time-step
                    inputs_part1 = one_hot_move
                    # inputs_part2 = tf.one_hot(np.full(self.numsamples, protein[n-1]-1), depth=2, dtype=tf.float64)    #! may not need to use tf.full if tf.concat takes care of broadcasting
                    inputs_part2 = tf.one_hot(np.full(self.numsamples, protein[n]-1), depth=2, dtype=tf.float64)
                    inputs = tf.concat((inputs_part1, inputs_part2), axis=1)
                    # inputs_part4 = tf.concat(([tf.math.divide(position, (2*self.N))], [tf.math.floormod(position,(2*self.N))]), axis=0)
                    # inputs_part4 = tf.transpose(inputs_part4, perm=[1,0])
                    # inputs = tf.concat((inputs_part1, inputs_part2, inputs_part3, inputs_part4), axis=1)
                    
                    # prepare mask for next iteration of sampling
                    mask_left = self.checker_func(previous_positions, position+offset[0])
                    mask_right = self.checker_func(previous_positions, position+offset[1])
                    mask_up = self.checker_func(previous_positions, position+offset[2])
                    mask_down = self.checker_func(previous_positions, position+offset[3])
                    # mask = tf.reshape(tf.concat([[mask_left],[mask_right],[mask_up], [mask_down]], axis = 1), shape = [self.numsamples, -1])
                    mask = tf.stack([mask_left, mask_right, mask_up, mask_down], axis=1)
                    # masks.append(mask)

        log_probs = tf.transpose(tf.stack(values=log_probs, axis=2), perm=[0,2,1])      # (numsamples, N-1, 4)
        moves = tf.stack(values=samples, axis=1)    # (numsamples, N-1)
        positions = tf.cast(tf.stack(values=previous_positions, axis=1), dtype=tf.int64)
        self.samples = moves    # (numsamples, N-1)
        one_hot_samples = tf.one_hot(self.samples, depth=self.outputdim, dtype=tf.float64)      # (numsamples, N-1, 4)
        self.log_probs = tf.reduce_sum(tf.multiply(log_probs, one_hot_samples), axis=[1,2])     # (numsamples,)

        return self.samples, self.log_probs, positions

    def log_probability(self, samples, protein):
        """
        description
        """
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            log_probs = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                self.numsamples = samples.shape[0]
                inputs = tf.zeros((self.numsamples, self.inputdim), dtype=tf.float64) #Feed the table b in tf.
                #Initial input to feed to the rnn
                rnn_states = []
                for i in range(self.numlayers):
                    for n in range(self.N):
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        # rnn_states.append(self.rnn[0][i].zero_state(self.numsamples, dtype=tf.float64)) #Initialize the RNN hidden state
                        zero_state = np.zeros((self.numsamples,self.numunits), dtype=np.float64)
                        protein_onehot = np.zeros(2, dtype=np.float64)
                        protein_onehot[protein[0]-1] = 1
                        zero_state[:, :2] = protein_onehot
                        zero_state = tf.convert_to_tensor(zero_state)
                        rnn_states.append(zero_state) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)
                previous_positions = []
                position = tf.constant([[2*self.N**2 + self.N] for _ in range(self.numsamples)], tf.float64)      # initial position at (N,N)
                position = tf.reshape(position, shape=[self.numsamples])
                # the effect of an L, R, U, D move on a position
                offset = np.array([-1, 1, -2*self.N, 2*self.N], np.float64)
                mask = tf.constant([1,1,1,1], dtype = tf.float64)
                # SAMPLING
                for n in range(1, self.N):
                    rnn_output = inputs
                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[n-1][i](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[n-1][i](rnn_output, rnn_states[i])

                    log_probs_nth = self.dense[n-1](rnn_output)
                    log_probs_nth = self.projection_identity(log_probs_nth, mask)
                    log_probs.append(log_probs_nth)

                    one_hot_move = tf.reshape(tf.one_hot(samples[:,n-1], depth=4, dtype=tf.float64), shape=[self.numsamples, -1])
                    position = position + tf.reduce_sum(one_hot_move*offset, axis=1)
                    # print(position)
                    previous_positions.append(position)
                    # INPUT COMPONENTS FOR NEXT RNN CELL
                    inputs_part1 = one_hot_move
                    # inputs_part2 = tf.one_hot(np.full(self.numsamples, protein[n-1]-1), depth=2, dtype=tf.float64) # may not to use tf.full if tf.concat takes care of broadcasting
                    inputs_part2 = tf.one_hot(np.full(self.numsamples, protein[n]-1), depth=2, dtype=tf.float64)
                    inputs = tf.concat((inputs_part1, inputs_part2), axis=1)
                    # inputs_part4 = tf.concat(([tf.math.divide(position, (2*self.N))], [tf.math.floormod(position,(2*self.N))]), axis = 0)    # normalized coordinates
                    # inputs_part4 = tf.transpose(inputs_part4, perm=[1,0])
                    # inputs = tf.concat((inputs_part1, inputs_part2, inputs_part3, inputs_part4), axis=1)
                    # !debug projection
                    mask_left = self.checker_func(previous_positions, position+offset[0])
                    mask_right = self.checker_func(previous_positions, position+offset[1])
                    mask_up = self.checker_func(previous_positions, position+offset[2])
                    mask_down = self.checker_func(previous_positions, position+offset[3])
                    mask = tf.reshape(tf.concat([[mask_left],[mask_right],[mask_up], [mask_down]], axis=1), shape=[self.numsamples, -1])

        log_probs = tf.transpose(tf.stack(values=log_probs, axis=2), perm=[0,2,1])
        one_hot_samples = tf.one_hot(samples, depth=self.outputdim, dtype=tf.float64)
        self.log_probs = tf.reduce_sum(tf.multiply(log_probs, one_hot_samples), axis=[1,2])

        return self.log_probs
    
def create_fold_grid(positions, protein):
    numsamples, N = positions.shape[0], positions.shape[1]
    # 2N+1 by 2N grid. Extra row to ensure no index error occurs when accessing 4-directional neighbors
    grid = np.zeros(shape=(numsamples, (2*N)**2 + 2*N), dtype=np.float32)
    indices = np.arange(numsamples).reshape(numsamples, 1)
    grid[indices, positions] = protein
    return grid

def energyv2(grid, positions, protein):
    numsamples, N = positions.shape[0], positions.shape[1]
    offset = np.array([-1, 1, -2*N, 2*N], np.float64)
    indices = np.arange(numsamples).reshape(numsamples, 1)
    ones = np.ones((numsamples, 4), np.float64)
    energies = np.zeros(numsamples)
    # prepare auxiliary positions
    positions_aux = np.zeros((numsamples, N+2), np.int32)
    positions_aux[:,0], positions_aux[:,-1], positions_aux[:,1:-1] = positions[:,0], positions[:,-1], positions[:,:]
    # get mask
    for i in range(1, positions_aux.shape[1]-1):
        if protein[i-1] == 1:
            # get neighborhood mask for unoccupied positions
            position_neighbors = (positions_aux[:,i].reshape((numsamples,1)) + offset).astype(int)
            # mask = np.where((position_neighbors[:,:,None] == positions_aux[:,[i-1,i+1]][:,None,:]).any(axis=-1), 0., 1.)
            mask = np.invert((position_neighbors[:,:,None] == positions_aux[:,[i-1,i+1]][:,None,:]).any(axis=-1)).astype(float) # faster than np.where
            # get neighborhood mask for H beads
            HP_neighbors = grid[indices,position_neighbors]
            H_neighbors = (grid[indices,position_neighbors] == ones).astype(float)
            energies += -np.sum(mask*H_neighbors, axis=1)

    return np.where(energies/2 == energies//2, energies/2, 0)

# ignore this function
def fold_is_valid(sample_positions):
    N = sample_positions.shape[0]
    _, unique_indices = np.unique(sample_positions, return_index=True)
    if unique_indices.shape[0] == N:    # valid fold
        return 1
    else:                               # invalid fold
        return 0

def fold_validity(sample_positions):
    N = sample_positions.shape[0]
    _, unique_indices = np.unique(sample_positions, return_index=True)
    # if all positions are unique then length of unique_indices remains same; valid fold (return 0)
    if unique_indices.shape[0] == N:
        return 0
    unique_positions = np.zeros_like(sample_positions, dtype=np.int32)
    unique_positions[unique_indices] = 1    # a unique position gets 1, duplicate gets 0
    duplicate_index = np.where(unique_positions == 0)[0][0]     # get the index of only the first duplicate position
    # check if a free position could have been sampled instead of the duplicate position
    # count is the number of the 4 possible positions that had already been sampled before the duplicate occured.
    # count=1 means three positions are free to sample ---> count=4 means there is no free position to sample
    offset = np.array([-1, 1, -2*N, 2*N], np.int64)
    count = np.sum(np.isin(sample_positions[:duplicate_index-1], offset + sample_positions[duplicate_index-1]).astype(int))
    if count == 4:  # dead-end scenario where all 4 possible positions are filled; invalid fold (return 1)
        return 1
    else:           # scenario where the model did not sample a free position; bug (return 2)
        return 2

def moves_to_positions(sample):
    N = sample.shape[0] + 1
    positions = np.zeros(N, dtype=np.int64)
    offset = np.array([-1, 1, -2*N, 2*N], np.int64)
    positions[0] = 2*N**2 + N
    for i in range(1, N):
        positions[i] = positions[i-1] + offset[sample[i-1]]
    return positions

# function to find all the 8 symmetries of a sample
def get_sample_symmetries(sample):
    coeffs = [[0, 1, 0, 0], [1, -23/6, 7/2, -2/3], [0, -5/6, 5/2, -2/3], [2, 23/6, -7/2, 2/3],
                    [2, 17/3, -6, 4/3], [3, -1, 0, 0], [1, -17/3, 6, -4/3], [3, 5/6, -5/2, 2/3]]
    symmetries = []
    for i in range(8):
        symmetries.append(np.round(coeffs[i][0]*np.power(sample, 0) + coeffs[i][1]*np.power(sample, 1) + coeffs[i][2]*np.power(sample, 2) + coeffs[i][3]*np.power(sample, 3)))
    return symmetries

def get_all_gs_folds(samples, energies):
    # get unique gs samples generated by model
    gs_index = np.where(energies == np.min(energies))
    gs_samples = samples[gs_index]
    samples_unqiue = np.unique(gs_samples, axis=0)
    # get true unique gs
    unique_count = 0
    gs_unique = dict()
    for i, sample in enumerate(samples_unqiue):
        if i == 0:
            sample_symmetries = get_sample_symmetries(sample)
            gs_unique[unique_count] = sample_symmetries
            unique_count += 1
        else:   # check if the gs sample already exists
            unique = True
            for gs in gs_unique:
                for j in range(8):
                    if np.array_equal(sample, gs_unique[gs][j]):
                        unique = False
                        break
                if not unique:
                    break
            if unique:      # if unique gs found, add its symmetries to the dictionary
                sample_symmetries = get_sample_symmetries(sample)
                gs_unique[unique_count] = sample_symmetries
                unique_count += 1
    return gs_unique


def groundstate_info(n_anneal, energies, positions):
    # validity = []
    # for i in range(n_warmup + n_anneal*5):
    #     for j in range(100):
    #         validity.append(fold_validity(positions[i][j]))
    # validity = np.array(validity)
    minE = np.min(np.array(energies))
    minE_indices = np.where(np.array(energies) == minE)
    print("gs energy: ", minE)
    print("gs sample count: ", len(minE_indices[0]))

    gs_samples = []
    count = 0
    for i in range(len(minE_indices[0])):
        row = minE_indices[0][i]
        col = minE_indices[1][i]
        # count += validity[row*100+col]
        # if validity[row*100+col] == 0:
        gs_samples.append(positions[row][col])
    # print("invalid gs count", count)
    gs_samples = np.array(gs_samples)
    gs_unique = np.unique(gs_samples, axis=0)
    for i in range(gs_unique.shape[0]):
        print(gs_unique[i])
    print("unique gs count", gs_unique.shape[0])

# get all distinct gs folds (excluding symmetries)
def get_unique_gs_folds(samples, energies):
    all_gs_folds = get_all_gs_folds(samples, energies)
    # remove symmetries
    unique_folds = []
    for i in range(len(all_gs_folds)):
        unique_folds.append(all_gs_folds[i][0])
    return np.array(unique_folds)

"""
VCA Class
"""
class vca:

    def __init__(self, N, protein, n_warmup, n_anneal, n_train, T0, annealer, inv_exp, ckpt_path, seed=111):
        """
        N - len(protein) - 1
        protein - HP sequence; list of int
        n_warmup - number of warmup iterations
        n_anneal - number of annaeling steps
        n_train - number of gradient/training steps for each annealing step
        T0 - initial temperature
        annealer - choose between linear annealing schedule vs. inverse
        inv_exp -   regulate the speed of the scheduler
                    if inverse annealing schedule, then T_(i+1) = T0 / pow(T_(i), 1/inv_exp)
        """
        tf.compat.v1.reset_default_graph()
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
        self.ckpt_path = ckpt_path
        
        # VCA hyperparameters
        self.num_warmup = n_warmup
        self.num_anneal = n_anneal
        self.num_train = n_train
        self.T0 = T0
        self.annealer = annealer
        self.inv_exp = inv_exp
        
        # RNN architecture hyperparameters
        self.N = N
        self.numlayers = np.int32(np.ceil(np.log2(N)))    # number of layers in the Dilated RNN architecture set to log2(N)
        # self.numlayers = 1                # one-dimensional RNN
        self.numunits = 50                  # size of RNN hidden units
        self.units = [self.numunits]*self.numlayers # defines the number of hidden units at each layer (for Dilated)
        self.numsamples = 200               # number of training samples, M
        self.lr = np.float64(1e-4)          # learning rate
        self.activation_function = tf.nn.tanh
        self.protein = protein
        
        # initialize RNN
        self.DRNN = DilatedRNN(N,units=self.units,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation=self.activation_function, seed=seed) #contains the graph with the RNNs

    def test_run(self):
        
        # building computation graph
        with tf.compat.v1.variable_scope(self.DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.DRNN.graph.as_default():

                global_step = tf.Variable(0, trainable=False)
                learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
                learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

                # tensorflow placeholders
                Eloc = tf.compat.v1.placeholder(dtype=tf.float64,shape=[self.numsamples])
                sampleplaceholder_forgrad = tf.compat.v1.placeholder(dtype=tf.int32, shape=[self.numsamples, self.N-1])
                log_probs_forgrad = self.DRNN.log_probability(sampleplaceholder_forgrad, protein=self.protein)
                outputs = self.DRNN.sample(numsamples=self.numsamples, protein=self.protein)
                T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

                # Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
                Floc = Eloc + T_placeholder*log_probs_forgrad
                # include the baseline
                cost = tf.reduce_mean(tf.multiply(log_probs_forgrad, tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

                # define optimizer
                gradients, variables = zip(*optimizer.compute_gradients(cost))
                optstep = optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)
                
                # GRADIENT CLIPPING
                # gradient_clipvalue = 1.0
                # gradients, variables = zip(*optimizer.compute_gradients(cost))
                # clipped_gradients = [tf.clip_by_value(g, -gradient_clipvalue, gradient_clipvalue) for g in gradients]
                # optstep = optimizer.apply_gradients(zip(clipped_gradients,variables),global_step=global_step)
                
                # GRADIENT NORMALIZATION
                # gradients, variables = zip(*optimizer.compute_gradients(cost))
                # gradients_normalized = []
                # for i in range(len(gradients)):
                #     random_numbers = tf.compat.v1.random_uniform(dtype=tf.float64,shape=tf.shape(gradients[i]))
                #     gradients_normalized.append(tf.sign(gradients[i])*random_numbers)
                # optstep=optimizer.apply_gradients(zip(gradients_normalized,variables),global_step=global_step)

                # tensorflow saver to checkpoint
                saver = tf.compat.v1.train.Saver()
                # for initialization
                init = tf.compat.v1.global_variables_initializer()

        # starting session
        # GPU management
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.compat.v1.Session(graph=self.DRNN.graph, config=config)
        sess.run(init)

        with tf.compat.v1.variable_scope(self.DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.DRNN.graph.as_default():
                
                # WARMUP loop - training steps for self.num_warmup steps at fixed temperature T0
                penalty = np.array([0, 0])
                T = self.T0
                for it in range(self.num_warmup):
                    # start the timer
                    if it == 0:
                        start = time.time()
                    samples, log_probs, positions = sess.run(outputs)
                    # check validity of fold
                    validity = np.zeros(self.numsamples, dtype=np.int64)
                    for i in range(positions.shape[0]):
                        validity[i] = fold_is_valid(positions[i])
                    # get grid and energy of fold
                    grids = create_fold_grid(positions, self.protein)
                    energies = energyv2(grids, positions, self.protein)
                    energies = (energies*validity + penalty[validity])          # penalize any fold that breaks SAW
                    
                    # printable metrics during the warm up stage
                    # compute average energy
                    meanE = np.mean(energies)
                    varE = np.var(energies)
                    # compute average free energy
                    meanF = np.mean(energies + T*log_probs)
                    varF = np.var(energies + T*log_probs)

                    # gradient step
                    sess.run(optstep,feed_dict={Eloc:energies, sampleplaceholder_forgrad:samples, learningrate_placeholder:self.lr, T_placeholder:T})

                # ANNEALING loop - core training with annealed/decaying temperature
                samples_history = np.zeros((self.num_anneal, self.num_train, self.numsamples, self.N-1))
                logprobs_history = np.zeros((self.num_anneal, self.num_train, self.numsamples))
                energies_history = np.zeros((self.num_anneal, self.num_train, self.numsamples))
                # positions_history = np.zeros((self.num_anneal, self.num_train, self.numsamples, self.N))
                temperatures = np.zeros(self.num_anneal)
                minE = 0
                start = time.time()
                for it0 in range(self.num_anneal):
                    # cool temperature based on annealer
                    if self.annealer == 'linear':
                        T = self.T0*(1-it0/self.num_anneal)                 # linear schedule
                    else:
                        T = self.T0 / np.power(it0+1, 1/self.inv_exp)       # inverse schedule, moderated by self.inv_exp
                    temperatures[it0] = T
                    # training loop - perform self.num_train gradient steps at every temperature/annealing step
                    for it1 in range(self.num_train):
                        samples, log_probs, positions = sess.run(outputs)
                        samples_history[it0,it1] = samples
                        logprobs_history[it0,it1] = log_probs
                        
                        # check validity
                        validity = np.zeros(self.numsamples, dtype=np.int64)
                        for i in range(positions.shape[0]):
                            validity[i] = fold_is_valid(positions[i])
                        # get energy
                        grids = create_fold_grid(positions, self.protein)
                        energies = energyv2(grids, positions, self.protein)
                        energies = energies * validity + penalty[validity]
                        
                        # track the min energy thus far
                        if np.min(energies) < minE:
                            minE = np.min(energies)
                        energies_history[it0,it1] = energies
                        
                        # compute average energy
                        meanE = np.mean(energies)
                        varE = np.var(energies)
                        # compute average free energy
                        meanF = np.mean(energies + T*log_probs)
                        varF = np.var(energies + T*log_probs)
                        
                        # print metrics
                        if it1 % 5 == 0:
                            print('ANNEALING PHASE')
                            print('min(E): {0}, mean(E): {1}, mean(F): {2}, var(E): {3}, var(F): {4}, #Annealing step {5}'.format(minE, meanE, meanF, varE, varF, it0))
                            print(minE)
                            print("Temperature: ", T)
                            print("Elapsed time: ", time.time()-start, " seconds")
                        
                        # gradient step
                        sess.run(optstep,feed_dict={Eloc:energies, sampleplaceholder_forgrad:samples, learningrate_placeholder:self.lr, T_placeholder:T})

                # # post-cooling - generate 500,000 folds after complete training
                # if it0 == self.num_anneal-1:
                #     # save model post-cooling
                #     # saver.save(sess, os.path.join(self.ckpt_path, "model.ckpt"))
                #     n_samples = 500000//self.numsamples
                #     samples_cooled = np.zeros((n_samples, self.numsamples, self.N-1))
                #     energies_cooled = np.zeros((n_samples, self.numsamples))
                #     logprobs_cooled = np.zeros((n_samples, self.numsamples))
                #     for it2 in range(n_samples):
                #         samples, log_probs, positions = sess.run(outputs)
                #         validity = np.zeros(self.numsamples, dtype=np.int64)
                #         for i in range(positions.shape[0]):
                #             validity[i] = fold_is_valid(positions[i])
                #         grids = create_fold_grid(positions, self.protein)
                #         energies = energyv2(grids, positions, self.protein)
                #         energies = energies * validity + penalty[validity]
                #         energies_cooled[it2] = energies
                #         samples_cooled[it2] = samples
                    
                    # return training data: energies_history, samples_history, logprobs_history
                    # return post-training data: energies_cooled, samples_cooled, logprobs_cooled
                    # return annealing schedule: temperatures
                    return energies_history, samples_history, logprobs_history, temperatures


