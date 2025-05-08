import nmrglue as ng
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from airPLS import airPLS

def read_bruker_h(nmr_path, bRaw = False, bMinMaxScale = False):
    nmr_path = os.path.normpath(nmr_path)
    if bRaw:
        dic, fid = ng.fileio.bruker.read(f'{nmr_path}/1')
        zero_fill_size = dic['acqus']['TD']
        fid = ng.bruker.remove_digital_filter(dic, fid)
        fid = ng.proc_base.zf_size(fid, zero_fill_size)
        fid = ng.proc_base.fft(fid)
        fid = ng.proc_autophase.autops(fid, 'acme', **{'disp':False})
        fid = ng.proc_base.rev(fid) 
    else:
        dic, fid = ng.fileio.bruker.read_pdata(f'{nmr_path}/1/pdata/1')
        zero_fill_size = dic['acqus']['TD']
    if bMinMaxScale:
        fid = fid / np.max(fid)
    offset = (float(dic['acqus']['SW']) / 2) - (float(dic['acqus']['O1']) / float(dic['acqus']['BF1']))
    start = float(dic['acqus']['SW']) - offset
    end = -offset
    step = float(dic['acqus']['SW']) / zero_fill_size
    ppms = np.arange(start, end, -step)[:zero_fill_size]
    baseline = airPLS(fid, lambda_=100, porder=1, itermax=15)
    fid = fid - baseline
    v = np.max(np.where(np.round(ppms, 3) == 12.500))
    b = v + 39042
    return {'name': nmr_path.split(os.sep)[-1], 'ppm': ppms[v:b], 'fid': fid[v:b], 'bRaw': bRaw}


def read_bruker_hs(data_folder, bRaw, bMinMaxScale, bDict):
    if bDict:
        spectra = {}
    else:
        spectra = []
    for name in tqdm(os.listdir(data_folder), desc="Read Bruker H-NMR files"):
        nmr_path = os.path.normpath(os.path.join(data_folder, name))
        s = read_bruker_h(nmr_path, bRaw, bMinMaxScale)
        if bDict:
            spectra[s['name']] = s
        else:
            spectra.append(s)
    return spectra


def plot_h(spectrum):
    fig = plt.figure(figsize=(12,4)) 
    ax = fig.add_subplot(1,1,1) 
    ax.plot(spectrum['ppm'], spectrum['fid'], color='red', linewidth=4.2) 
    ax.set_xlim((spectrum['ppm'][0], spectrum['ppm'][-1]))
    ax.set_xlabel('ppm')
    ax.set_ylabel('Intensity')
    ax.set_title(spectrum['name'])


def plot_hs(spectra):
    fig = plt.figure(figsize=(12,4)) 
    ax = fig.add_subplot(1,1,1) 
    for s in spectra:
        ax.plot(s['ppm'], s['fid'], label=s['name']) 
    ax.set_xlim((spectra[0]['ppm'][0], spectra[0]['ppm'][-1]))
    ax.set_xlabel('ppm')
    ax.set_ylabel('Intensity')
    ax.legend()
    

if __name__=="__main__":
    spectrum = read_bruker_h('D:/workspace/DeepNMR_code/data/flavor_standards/Benzyl acetate', False, True)
    plot_h(spectrum)
    spectra = read_bruker_hs('D:/workspace/DeepNMR_code/data/flavor_standards', False, True, False)
    #plot_hs(spectra)

    