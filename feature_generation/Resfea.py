import numpy as np
import linecache
import math
import pickle
import os
from sklearn.preprocessing import MinMaxScaler


# Data normalization
def normalize(value):
    a = 1 + math.exp(0-value)
    b = 1 / a
    return b

def PSSMfeature(path, uniprot_id):
    pssmfilelines = linecache.getlines(path + '/' + uniprot_id + '/' + uniprot_id + '.prf')
    protein_pssm = []
    length = len(pssmfilelines)
    for i in range(3, length):
        eachres_pssm = []
        line = pssmfilelines[i].split()
        if len(line) == 0:
            break
        else:
            for j in range(2, 22):
                eachres_pssm.append(normalize(int(line[j])))
            protein_pssm.append(eachres_pssm)
    return np.array(protein_pssm, dtype='float32')

def ssfeature(path, uniprot_id):
    protein_ss = []
    filelines = linecache.getlines(path + '/' + uniprot_id + '/' + uniprot_id + '.ss2')
    length = len(filelines)
    for i in range(2, length):
        eachres_ss = []
        eachres_ss.append(float(filelines[i].split()[3]))
        eachres_ss.append(float(filelines[i].split()[4]))
        eachres_ss.append(float(filelines[i].split()[5]))
        protein_ss.append(eachres_ss)
    return np.array(protein_ss, dtype='float32')

def rsafeature(path, uniprot_id):
    protein_rsa = []
    filelines = linecache.getlines(path + '/' + uniprot_id + '/' + uniprot_id + '.a3')
    length = len(filelines)
    for i in range(2, length):
        eachres_rsa = []
        eachres_rsa.append(float(filelines[i].split()[3]))
        eachres_rsa.append(float(filelines[i].split()[4]))
        eachres_rsa.append(float(filelines[i].split()[5]))
        protein_rsa.append(eachres_rsa)
    return np.array(protein_rsa, dtype='float32')

def embeddingfea(path, uniprot_id):
    file = path + '/' + uniprot_id + '/' + uniprot_id + '.npy'
    protein_embedding = np.load(file)
    scaler = MinMaxScaler()
    scaler_embedding = scaler.fit_transform(protein_embedding)
    return np.array(scaler_embedding, dtype='float32')

if __name__ == '__main__':
    """
    Generate feature for each protein
    MSA + LM + SS
    """
    path = '../feature/test'
    proteins = os.listdir(path)
    print('Generate feature for each protein.....')
    for protein in proteins:
        print(protein)
        protein_path = path + '/' + protein + '/'
        pssm = np.load(protein_path + protein + '_pssm.npy')
        ss = np.load(protein_path + protein + '_ss.npy')
        rsa = np.load(protein_path + protein + '_rsa.npy')
        embedding = np.load(protein_path + protein + '_embedding.npy')
        snb_pssm = np.load(protein_path + protein + '_snb-pssm.npy')
        feature = np.concatenate((pssm, ss, rsa, embedding, snb_pssm), axis=1).astype('float32')
        np.save(protein_path + protein + '_all_fea.npy', feature)
    print('Done!')
