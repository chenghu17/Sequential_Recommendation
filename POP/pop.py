import pandas as pd
import numpy as np
import random


class statistic_data():

    def __init__(self, datatype):
        self.datatype = datatype
        self.train_dataset = './data/' + self.datatype + '/' + self.datatype + '_train_dataset.csv'
        self.test_dataset = './data/' + self.datatype + '/' + self.datatype + '_test_dataset.csv'
        self.train_items = dict()
        self.test_real_items = []

    def analyze_train(self):
        data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in data.values:
            if is_first_line:
                is_first_line = 0
                continue
            sessions = [s for s in line[1].split('@')]
            size = len(sessions)
            for j in range(size):
                current_session = [int(it) for it in sessions[j].split(':')]
                for item in current_session:
                    if item not in self.train_items.keys():
                        self.train_items[item] = 1
                    else:
                        self.train_items[item] += 1

    def analyze_test(self):
        data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        sub_index = self.shuffle(len(data.values))
        data = data.values[sub_index]
        for line in data:
            current_session = [int(i) for i in line[1].split(':')]
            item = random.choice(current_session)
            self.test_real_items.append(int(item))

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index


class pop():

    def __init__(self, datatype):
        self.datatype = datatype
        sd = statistic_data(datatype)
        sd.analyze_train()
        sd.analyze_test()
        self.items_pop = sd.train_items
        self.test_real_items = sd.test_real_items

    def P_k(self, pop_k):
        right_pre = 0
        record_num = len(self.test_real_items)
        for item in self.test_real_items:
            if item in pop_k:
                right_pre += 1
        return right_pre / record_num
        pass

    def MRR_k(self, pop_k):
        MRR_rate = 0
        record_num = len(self.test_real_items)
        for item in self.test_real_items:
            if item in pop_k:
                rank = pop_k.index(item)
                MRR_rate += 1 / (rank + 1)
        return MRR_rate / record_num

    def metric(self):
        result = sorted(self.items_pop.items(), key=lambda item: item[1])
        # pop_10 = [i[0] for i in result[:10]]
        # pop_20 = [i[0] for i in result[:20]]
        pop_50 = [item[0] for item in result[:50]]

        # P_10 = self.P_k(pop_10)
        # MRR_10 = self.MRR_k(pop_10)
        #
        # P_20 = self.P_k(pop_20)
        # MRR_20 = self.MRR_k(pop_20)

        P_50 = self.P_k(pop_50)
        MRR_50 = self.MRR_k(pop_50)

        # print('P_10:', P_10, ',MRR_10:', MRR_10, '\n')
        # print('P_20:', P_20, ',MRR_20:', MRR_20, '\n')
        # print('P_50:', P_50, ',MRR_50:', MRR_50)
        return P_50, MRR_50


if __name__ == '__main__':
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    index = 5
    itera = 50
    P_50 = 0
    MRR_50 = 0
    for i in range(itera):
        model = pop(type[index])
        P, MRR = model.metric()
        P_50 += P
        MRR_50 += MRR
    print('P_50:', P_50 / itera, ',MRR_50:', MRR_50 / itera)
