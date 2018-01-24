# Copy Right Kairos03 2018. All Right Reserved.

from sklearn.model_selection import train_test_split
import numpy as np

from data import process


class Dataset:
    def __init__(self, batch_size, data, label, is_shuffle=False, is_valid=False):
        self.data = data
        self.label = label

        self.valid_data = None
        self.valid_label = None

        self.data_size = self.data.shape[0]
        print(self.data_size)
        self.batch_size = batch_size
        self.total_batch = int(self.data_size / self.batch_size) + 1
        self.batch_cnt = 0

        self.is_shuffle = is_shuffle
        self.is_valid = is_valid

        if is_valid:
            self.data, self.valid_data, self.label, self.valid_label = train_test_split(self.data,
                                                                                        self.label,
                                                                                        test_size=0.33,
                                                                                        random_state=486)

            self.data_size = self.data.shape[0]
            self.valid_size = self.valid_data.shape[0]

            self.total_batch = int(self.data_size / self.batch_size) + 1
            self.valid_total_batch = int(self.valid_size / self.batch_size) + 1

    def next_batch(self, seed, valid_set=False):

        if valid_set:
            data = self.valid_data
            label = self.valid_label
            total_batch = self.valid_total_batch
        else:
            data = self.data
            label = self.label
            total_batch = self.total_batch

        # shuffle
        if self.is_shuffle and self.batch_cnt == 0:
            np.random.seed(seed)
            np.random.shuffle(data)
            np.random.seed(seed)
            np.random.shuffle(label)

        start = self.batch_cnt * self.batch_size
        self.batch_cnt += 1

        if self.batch_cnt == total_batch:
            end = None
        else:
            end = self.batch_cnt * self.batch_size

        xs = data[start:end]
        ys = label[start:end]

        if self.batch_cnt >= total_batch:
            self.batch_cnt = 0

        return xs, ys

    # def get_test(self):
    #     return self.test[:][0], self.test[:][1]


def get_dataset(batch_size, data, label, is_shuffle, is_valid):
    return Dataset(batch_size=batch_size, data=data, label=label, is_shuffle=is_shuffle, is_valid=is_valid)


if __name__ == '__main__':

    # deprecated
    # x, y, idx = load_data()

    x, y = process.load_image_train_dataset()
    RANDOM_SEED = 128

    dd = get_dataset(500, x, y, is_shuffle=False, is_valid=True)

    print(dd.valid_data.shape)
    print(dd.data.shape)

    for e in range(2):
        for b in range(dd.total_batch):
            xss, yss = dd.next_batch(RANDOM_SEED, valid_set=False)
            print(e, b, xss.shape, yss.shape)

    for e in range(2):
        for b in range(dd.valid_total_batch):
            xss, yss = dd.next_batch(RANDOM_SEED, valid_set=True)
            print(e, b, xss.shape, yss.shape)
