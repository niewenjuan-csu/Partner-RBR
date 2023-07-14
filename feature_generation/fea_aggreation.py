import numpy as np
import pickle
import joblib
import random
from sklearn.model_selection import StratifiedKFold
import gc

RANDOM_SEED = 42

def savefile(filetosave, data):
    with open(filetosave, 'wb') as fo:
        joblib.dump(data, fo)


# padding with zero
def appendzero(windowsize, featurevec):
    tempfea = featurevec
    appendnum = int((windowsize + 1) / 2)
    append_length = len(featurevec[0])
    for i in range(1, appendnum):
        tempfea.insert(0, [0]*append_length)
        tempfea.append([0]*append_length)
    return tempfea

def combine(seqlength, feature, windowsize):
    neighnum = int((windowsize - 1) / 2)
    combinefea = []
    for i in range(0+neighnum, seqlength+neighnum):
        window_fea = []
        for a in range(i - neighnum, i + neighnum + 1):
            window_fea.append(feature[a])
        combinefea.append(window_fea)
    return combinefea

# for one protein
def featurecombine(uniprot_id, pssm, ss, rsa, embedding, snb_pssm, windowsize=11):
    assert len(pssm) == len(ss) == len(rsa) == len(embedding) == len(snb_pssm)
    protein_length = len(embedding)
    protein_fea = []
    for i in range(protein_length):
        eachres_fea = []
        # msa-based feature
        eachres_fea.extend(list(pssm[i]))
        eachres_fea.extend(list(ss[i]))
        eachres_fea.extend(list(rsa[i]))
        # embedding-based feature
        eachres_fea.extend(list(embedding[i]))
        # structure-based feature
        eachres_fea.extend(list(snb_pssm[i]))
        # feature for each residue
        protein_fea.append(eachres_fea)
    # print(protein_fea)
    appendedfeature = appendzero(windowsize, protein_fea)
    combinefeature = combine(protein_length, appendedfeature, windowsize)
    return np.array(combinefeature)

# 获得训练集的残基水平的特征向量
def get_feature(path, file, protein_list, windowsize):
    f = open(file, 'r')
    data = f.readlines()
    # res_info stores the information of each residue
    res_info = []
    for i in range(len(data)):
        if data[i].startswith('>'):
            protein = data[i].lstrip('>').strip()
            if protein in protein_list:
                print(protein)
                protein_pssm = np.load(path+'/'+protein+'/'+protein+'_pssm.npy')
                protein_ss = np.load(path + '/' + protein + '/' + protein + '_ss.npy')
                protein_rsa = np.load(path + '/' + protein + '/' + protein + '_rsa.npy')
                protein_embedding = np.load(path + '/' + protein + '/' + protein + '_embedding.npy')
                protein_snb_pssm = np.load(path + '/' + protein + '/' + protein + '_snb-pssm.npy')
                # sequence-based feature
                protein_feature = np.array(featurecombine(protein, protein_pssm, protein_ss, protein_rsa,
                                                          protein_embedding,
                                                          protein_snb_pssm,
                                                          windowsize=windowsize), dtype='float32')
                # structure-based feature
                protein_neibor_feature = np.load(path + '/' + protein + '/' + protein + '_neibor.npy').astype('float32')
                sequence = data[i + 1].strip()
                protein_length = len(sequence)
                RNA_label = data[i + 2].strip()
                rRNA_label = data[i + 3].strip()
                tRNA_label = data[i + 4].strip()
                snRNA_label = data[i + 5].strip()
                mRNA_label = data[i + 6].strip()
                SRP_label = data[i + 7].strip()

                assert protein_length == len(protein_neibor_feature) == len(RNA_label) == len(rRNA_label) == len(tRNA_label) == len(snRNA_label) == len(mRNA_label) == len(SRP_label) == len(protein_feature)
                # generate label
                for j in range(protein_length):
                    label = [eval(rRNA_label[j]), eval(tRNA_label[j]), eval(snRNA_label[j]), eval(mRNA_label[j]), eval(SRP_label[j])]
                    if label.count(1) == 0:
                        multi_label = 0
                    else:
                        multi_label = label.index(1) + 1
                    multi_label = np.array(multi_label).astype('int32')

                    eachres_info = []
                    eachres_info.append(protein_feature[j])
                    eachres_info.append(protein_neibor_feature[j])
                    eachres_info.append(multi_label)
                    res_info.append(eachres_info)
    return res_info

def undersamper(data, chosen_num):
    np.random.seed(RANDOM_SEED)
    random.shuffle(data)
    sample_data = data[0: chosen_num]
    return sample_data


def oversamper(data, weight):
    np.random.seed(RANDOM_SEED)
    random.shuffle(data)
    sample_data = []
    for i in range(weight):
        sample_data.extend(data)
    np.random.seed(RANDOM_SEED)
    random.shuffle(sample_data)
    return sample_data


def get_class_index(data, targets):
    class_index = dict()
    for target in targets:
        class_index[target] = []
    for i in range(len(data)):
        if data[i][2] == 0:
            class_index['non'].append(i)
        if data[i][2] == 1:
            class_index['rRNA'].append(i)
        if data[i][2] == 2:
            class_index['tRNA'].append(i)
        if data[i][2] == 3:
            class_index['snRNA'].append(i)
        if data[i][2] == 4:
            class_index['mRNA'].append(i)
        if data[i][2] == 5:
            class_index['SRP'].append(i)
    return class_index


def get_balance_data(data, targets):
    class_index = get_class_index(data, targets)
    base_num = len(class_index['rRNA'])
    balance_data = []
    for target in targets:
        if target == 'non':
            sample_data = undersamper(class_index[target], base_num)
        else:
            sample_data = oversamper(class_index[target], base_num // len(class_index[target]))
        sample_data = np.array(sample_data)
        balance_data.extend(data[sample_data])
    return balance_data


if __name__ == '__main__':
    window_size = 11
    targets = ['non', 'rRNA', 'tRNA', 'snRNA', 'mRNA', 'SRP']

    """
    Generate profile for training dataset
    """
    path = '../feature/train'
    save_path = '../data'
    all_protein = []
    protein_label = []
    train_file = '../train.txt'
    f = open(train_file, 'r')
    data = f.readlines()
    for i in range(len(data)):
        if data[i].startswith('>'):
            protein = data[i].lstrip('>').strip()
            all_protein.append(protein)
            rRNA_label = data[i + 3].strip()
            tRNA_label = data[i + 4].strip()
            snRNA_label = data[i + 5].strip()
            mRNA_label = data[i + 6].strip()
            SRP_label = data[i + 7].strip()
            if rRNA_label.count('1') != 0:
                label = 1
            elif tRNA_label.count('1') != 0:
                label = 2
            elif snRNA_label.count('1') != 0:
                label = 3
            elif mRNA_label.count('1') != 0:
                label = 4
            elif SRP_label.count('1') != 0:
                label = 5
            else:
                label = 0
            protein_label.append(label)
    all_protein = np.array(all_protein)
    protein_label = np.array(protein_label)
    gc.collect()
    i = 1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(all_protein, protein_label):
        print('Fold%d' %(i))
        protein_to_train = all_protein[train_index]
        protein_to_valid = all_protein[test_index]
        train = get_feature(path, train_file, protein_to_train, window_size)
        valid = get_feature(path, train_file, protein_to_valid, window_size)
        # savefile(save_path+'/fold{:d}/train.pkl'.format(i), train)
        savefile(save_path+'/fold{:d}/valid.pkl'.format(i), valid)

        train = np.array(train)
        train_balance = get_balance_data(train, targets)
        savefile(save_path+'/fold{:d}/train_balance.pkl'.format(i), train_balance)

        i += 1
        gc.collect()

    print('Data for train, Done!')


    """
    Generation feature for test set
    """
    test_path = '../feature/test'
    test_file = '../test.txt'
    tf = open(test_file, 'r')
    tdata = tf.readlines()
    tprotein = []
    for i in range(len(tdata)):
        if tdata[i].startswith('>'):
            tprotein.append(tdata[i].lstrip('>').strip())
    for j in range(len(tprotein)):
        test = get_feature(test_path, test_file, tprotein[j], window_size)
        savefile('../case_study/test_protein/'+tprotein[j]+'.pickle', test)

    print('Data for test, Done!')
