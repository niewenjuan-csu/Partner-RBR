import numpy as np
import xlrd
import os
from sklearn.preprocessing import MinMaxScaler

def read_excel(file):
    resArray=[]
    data = xlrd.open_workbook(file)
    table = data.sheet_by_index(0)
    for i in range(table.nrows):
        line=table.row_values(i)
        resArray.append(line)
    resArray=np.array(resArray).astype('float32')
    sclaer = MinMaxScaler()
    resArray = sclaer.fit_transform(resArray)
    return resArray

if __name__ == '__main__':
    path = '../feature/test'
    proteins = os.listdir(path)
    print('Transform snb-pssm.xlsx to array to save...')
    for protein in proteins:
        print(protein)
        # P01327 has positional missing, need to process alone.
        if protein == 'P01327':
            continue
        else:
            protein_path = path + '/' + protein + '/'
            file = protein_path + protein + '_snb.xlsx'
            snb_pssm = read_excel(file)
            np.save(protein_path + protein + '_snb-pssm.npy', snb_pssm)
    print('Done!')

    # protein = 'P01327'
    # protein_path = path + '/' + protein + '/'
    # file = protein_path + protein + '_snb.xlsx'
    # snb_pssm = read_excel(file)
    # np.save(protein_path + protein + '_snb-pssm.npy', snb_pssm)


