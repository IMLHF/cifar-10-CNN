import pickle

def get_batch_use_tfdata():
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
