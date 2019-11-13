
import numpy as np
def batch(batch_size, dataset_length) :
    batch_index = np.arange(dataset_length)
    np.random.shuffle(batch_index)
    return batch_index[:batch_size]






if __name__ == '__main__':
    print(batch(1, 1))