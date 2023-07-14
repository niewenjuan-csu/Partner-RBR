import sklearn
import numpy as np
import torch
import pickle
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import cycle
from dml.utils import *
from dml.config import get_config
from torch_model.model import Bind
from dml.data_loader import get_train_loader, get_test_loader
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import joblib

def model_prediction(model, data):
    true = []
    prediction = []
    output_fea = []

    model.eval()
    for i, (x, neibor_x, label) in enumerate(data):
        x, neibor_x, label = Variable(x), Variable(neibor_x), Variable(label)
        true.extend(label.data.numpy())

        # forward pass
        outputs, fea = model(x, neibor_x)
        prediction.extend(outputs.detach().numpy())
        output_fea.extend(fea.detach().numpy())
        np.save('output_fea.npy', output_fea)

    return np.array(true), np.array(prediction)



def predict(config, data, model_path, model_list, targets, num_classes=6):
    all_prediction = []

    for model_name in model_list:
        model = Bind(num_classes)
        optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.momentum,
                              weight_decay=config.weight_decay, nesterov=config.nesterov)
        ckpt = torch.load(model_path + model_name)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        label, prediction = model_prediction(model, data)
        all_prediction.append(np.array(prediction))

    return all_prediction, label

if __name__ == '__main__':
    config, unparsed = get_config()

    test_file = './data/test.pickle'
    test_data_loader = get_test_loader(test_file, config.batch_size)

    test_path = './feature/test'
    test_protein_file = './test.txt'

    tf = open(test_protein_file, 'r')
    tdata = tf.readlines()
    protein_fasta = dict()
    for i in range(len(tdata)):
        if tdata[i].startswith('>'):
            tprotein = tdata[i].lstrip('>').strip()
            fasta_line = tdata[i+1].lstrip('>').strip()
            protein_fasta[tprotein] = fasta_line


    num_classes = 6
    targets = ['Non-RNA', 'rRNA', 'tRNA', 'snRNA', 'mRNA', 'SRP']

    model_path = './dml/5cv_ckpt/dml_50/'

    model_list = ['fold12_ckpt.pth', 'fold22_ckpt.pth', 'fold32_ckpt.pth',
                  'fold42_ckpt.pth', 'fold52_ckpt.pth',
                  'fold11_ckpt.pth', 'fold21_ckpt.pth', 'fold31_ckpt.pth',
                  'fold41_ckpt.pth', 'fold51_ckpt.pth']

    # 直接取平均
    all_prediction, true = predict(config, test_data_loader, model_path=model_path, model_list=model_list,
                                   targets=targets, num_classes=num_classes)

    pre = all_prediction[0] + all_prediction[1] + all_prediction[2] + all_prediction[3] + all_prediction[4] + \
          all_prediction[5] + all_prediction[6] + all_prediction[7] + all_prediction[8] + all_prediction[9]
    mean_prediction = pre / len(model_list)
    max_prediction, prediction_fpr5, prediction = metrics(mean_prediction, true, targets)

    # bind_num = 0
    # for i in range(len(max_prediction)):
    #     if max_prediction[i] > 0:
    #         bind_num += 1
    # print(bind_num / len(true))

    # sequence = protein_fasta[protein]
    # save_result(save_path, protein, sequence, mean_prediction, max_prediction, prediction)











