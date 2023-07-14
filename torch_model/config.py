class DefaultConfig(object):
    # 7 9 11 13 15 17 19 21
    windows_size = 11
    k = 8  # 残基的平均邻接残基个数
    neibor_num = k+1

    # node feature
    pssm_dim = 20
    ss_dim = 3
    rsa_dim = 3
    embed_dim = 1024
    snb_pssm_dim = 100
    dssp_dim = 19
    fea_dim = pssm_dim + ss_dim + rsa_dim + embed_dim + snb_pssm_dim

    # TextCNN
    kernel = [1, 3, 5, 7]
    # 32 64 128
    sequence_channle = 32
    structure_channel = 32

    # Dense Layer
    dropout_dense = 0.2








