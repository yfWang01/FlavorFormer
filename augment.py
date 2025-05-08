import random
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from readBruker import read_bruker_h, read_bruker_hs

def augment_positive(spectra, ids4p, low=0.2, high=1.0):
    no = random.randint(0, len(ids4p) - 1)  
    ratio = random.uniform(low, high)       
    shift = int(np.clip(np.random.normal(-5, 30), -50, 50))  
    x = ratio * np.roll(spectra[ids4p[-1]]['fid'], shift)  
    for i in range(no):
        ratio = random.uniform(low, high)
        shift = int(np.clip(np.random.normal(-5, 30), -50, 50))  
        x = x + ratio * np.roll(spectra[ids4p[i]]['fid'], shift)  
    return x
    
def augment_negative(spectra, ids4n, low=0.2, high=1.0):
    no = random.randint(1, len(ids4n) - 1)  
    x = np.zeros_like(spectra[0]['fid'])    
    for i in range(no):
        ratio = random.uniform(low, high) 
        shift = int(np.clip(np.random.normal(-5, 30), -50, 50))  
        x = x + ratio * np.roll(spectra[ids4n[i]]['fid'], shift)  
    return x

def data_augmentation(spectra, n, max_pc, noise_level = 0.001):
    p  = spectra[0]['ppm'].shape[0]   
    s  = len(spectra)
    Rp = np.zeros((n, p), dtype = np.float32)
    Sp = np.zeros((n, p), dtype = np.float32)
    Rn = np.zeros((n, p), dtype = np.float32)
    Sn = np.zeros((n, p), dtype = np.float32)
    for i in tqdm(range(n), desc="Data augmentation"):
        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)
        ids4p   = random.sample(range(0, s-1), max_pc)
        Rp[i, ] = spectra[ids4p[-1]]['fid'] + (n1-np.min(n1))*noise_level
        Sp[i, ] = augment_positive(spectra, ids4p) + (n2-np.min(n2))*noise_level
        ids4n   = random.sample(range(0, s-1), max_pc+1)
        Rn[i, ] = spectra[ids4n[-1]]['fid'] + (n3-np.min(n3))*noise_level
        Sn[i, ] = augment_negative(spectra, ids4n) + (n4-np.min(n4))*noise_level
    R = np.vstack((Rp, Rn))
    S = np.vstack((Sp, Sn))
    y  = np.concatenate((np.ones(n, dtype = np.float32), np.zeros(n, dtype = np.float32)), axis=None)
    R, S, y = shuffle(R, S, y)
    return {'R':R, 'S':S, 'y':y}

def augment_and_split_data(spectra, output_dir="data", augment_samples=10000, max_pc=5, noise_level=0.001):
    os.makedirs(output_dir, exist_ok=True)

    augmented_data = data_augmentation(spectra, augment_samples, max_pc, noise_level)
    R, S, y = augmented_data['R'], augmented_data['S'], augmented_data['y']

    R_train, R_temp, S_train, S_temp, y_train, y_temp = train_test_split(
        R, S, y, test_size=0.2, random_state=42
    )
    R_val, R_test, S_val, S_test, y_val, y_test = train_test_split(
        R_temp, S_temp, y_temp, test_size=0.5, random_state=42
    )

    datasets = {
        'train': {'R': R_train, 'S': S_train, 'y': y_train},
        'val': {'R': R_val, 'S': S_val, 'y': y_val},
        'test': {'R': R_test, 'S': S_test, 'y': y_test},
    }

    for name, data in datasets.items():
        file_path = os.path.join(output_dir, f"{name}_dataset.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"{name.capitalize()} dataset saved to {file_path}.")

def plot_augment(aug, ids):
    for i in ids:
        fig = plt.figure(figsize=(12,4)) 
        ax = fig.add_subplot(1,1,1) 
        ax.plot(aug['R'][i,], 'r', label = 'R') 
        ax.plot(aug['S'][i,], 'k', label = 'S') 
        ax.set_title(f"{aug['y'][i]}")
        ax.legend()

if __name__=="__main__":
    spectra = read_bruker_hs('data/flavor_standards', False, True, False)  
    output_dir = 'data/augmented_data/50000_samples_7_pc_0.001_noise_airPLS_12.5_0.57_shift30-5'
    augment_and_split_data(
        spectra,
        output_dir,
        augment_samples=50000,
        max_pc=7,
        noise_level=0.001
    )
