#%%
import numpy as np
from glob import glob
from scipy.interpolate import interp1d

# Global Variables
rmesh = np.linspace(0.06,6.00,100)
kspace = np.linspace(2,13.0,220)

SEED = 1202; rng = np.random.default_rng(SEED)

def seed_test():
    return rng.integers(low =0, high = 100)
def compute_coordination_number(gr,rmesh,rrange):
    gr = gr/10
    mask = (rmesh >= rrange[0]) & (rmesh <= rrange[1])
    r_selected = rmesh[mask]
    g_r_selected = gr[mask]
    dr = rmesh[1]-rmesh[0]
    # Compute coordination number using numerical integration
    CN = 4 * np.pi * np.trapz(g_r_selected * r_selected**2, r_selected,dr)
    return CN

def interpol(exafs):
    global kspace
    x, y = exafs.transpose()
    f1=interp1d(x, y, kind='cubic')
    test_real=f1(kspace)
    return test_real

def linear_combination(num_of_data_samples, data):
    # Number of datapoints to combine
    N = rng.integers(low = 1, high = 3, size = 1, endpoint = True)
    # Choose N samples randomly to combine
    C = rng.choice(num_of_data_samples, N, replace=False)
    weights = rng.dirichlet(np.ones(N), size=1)[0]
    selected_data = data[C]
    weighted_exafs = np.average([datapoint[0] for datapoint in selected_data], axis=0, weights=weights)
    #weighted_rdf = np.average([np.hstack((datapoint[1],datapoint[2],datapoint[3])) for datapoint in selected_data], axis=0, weights=weights)
    weighted_rdf = np.average([datapoint[1] for datapoint in selected_data], axis=0, weights=weights)
    return weighted_exafs, weighted_rdf

def generate_examples(batch_size, num_of_data_samples, data):
    examples = np.zeros((batch_size, 2), dtype=object) # change to 2 if only using 1 rdf
    for i in range(batch_size):
        examples[i] = linear_combination(num_of_data_samples, data)
    chi, rdf = zip(*examples)
    chi, rdf = np.array(chi), np.array(rdf)
    return chi, rdf

def data_generator(batch_size, num_of_data_samples, data):
    while True:
        x_batch_0, y_batch = generate_examples(batch_size, num_of_data_samples, data)

        # Reshaping exafs for convolution layer
        x_batch = x_batch_0.reshape(x_batch_0.shape[0], 1, x_batch_0.shape[1])
        
        # Noise level (Std of gaussian)
        n_level = rng.uniform(low=0.0, high=0.05)
        # Adding gaussian noise 
        noise = rng.normal(loc=0, scale=n_level, size=x_batch.shape)
        yield x_batch+ noise, y_batch # 
