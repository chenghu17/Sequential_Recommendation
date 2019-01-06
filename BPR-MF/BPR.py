# author : hucheng
# Standard Bayesian Personalized Ranking

import pandas as pd
import numpy as np
import random
import logging
from scipy.stats import logistic


class statistic_data():

    def __init__(self, datatype):
        self.datatype = datatype
        self.train_dataset = './data/' + self.datatype + '/' + self.datatype + '_train_dataset.csv'
        self.test_dataset = './data/' + self.datatype + '/' + self.datatype + '_test_dataset.csv'

        self.userNum = 0
        self.itemNum = 0

        self.train_users = set()  # 保存train中哪些用户打过分
        self.train_items = set()  # 保存train中出现过的item
        self.train_user_items = dict()  # 保存train中每个用户的打分情况

        self.test_real_items = []
        self.test_real_users = []

    def analyze_train(self):
        data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in data.values:
            if is_first_line:
                self.userNum = int(line[0])
                self.itemNum = int(line[1])
                is_first_line = 0
                continue
            userid = int(line[0])
            items = []
            sessions = [s for s in line[1].split('@')]
            size = len(sessions)
            for j in range(size):
                current_session = [int(it) for it in sessions[j].split(':')]
                items.extend(current_session)
                for item_id in current_session:
                    self.train_items.add(item_id)
            if userid in self.train_user_items.keys():
                self.train_user_items[userid].extend(items)
            else:
                self.train_user_items[userid] = items

    def analyze_test(self):
        data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        sub_index = self.shuffle(len(data.values))
        data = data.values[sub_index]
        for line in data:
            user_id = int(line[0])
            current_session = [int(i) for i in line[1].split(':')]
            item = random.choice(current_session)
            self.test_real_items.append(item)
            self.test_real_users.append(user_id)

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index


class BPR():
    def __init__(self, datatype, d, itera):

        self.data_type = datatype
        sd = statistic_data(self.data_type)
        sd.analyze_train()
        sd.analyze_test()

        self.train_users = sd.train_users
        self.train_items = sd.train_items
        self.train_user_items = sd.train_user_items

        self.test_real_users = sd.test_real_users
        self.test_real_items = sd.test_real_items

        self.userNum = sd.userNum
        self.itemNum = sd.itemNum

        self.dim = d
        self.itera = itera
        self.negNum = 1

        self.alpha = 0.01
        self.learning_rate = 0.001

        # 日志基本配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        fh = logging.FileHandler('log_' + self.data_type + '_d_' + str(self.dim), mode='a', encoding=None,
                                 delay=False)
        self.logger.addHandler(fh)

    def P_k(self, pre_top_k):
        right_pre = 0
        record_number = len(self.test_real_items)
        for i in range(record_number):
            real_item_id = self.test_real_items[i]
            if real_item_id in pre_top_k[i]:
                right_pre += 1
        return right_pre / record_number

    def MRR_K(self, pre_top_k):
        mrr = 0
        record_number = len(self.test_real_items)
        for i in range(record_number):
            real_item_id = self.test_real_items[i]
            if real_item_id in pre_top_k[i]:
                index = pre_top_k[i].index(real_item_id)
                mrr += 1 / (index + 1)
        return mrr / record_number

    def evalution(self):
        pre_top_k_50 = []
        for user_id in self.test_real_users:
            prediction = dict()
            Pu = self.userMat[user_id]
            for item_id in range(self.itemNum):
                Qi = self.itemMat[item_id]
                pre = np.dot(Pu, Qi)
                prediction[item_id] = pre

            # 对prediction取top_k，并返回index列表
            result = sorted(prediction.items(), key=lambda item: item[1])
            top_index_50 = [item[0] for item in result[:50]]
            pre_top_k_50.append(top_index_50)

        P_50 = self.P_k(pre_top_k_50)
        MRR_50 = self.MRR_K(pre_top_k_50)
        return P_50, MRR_50

    def standard_BPR(self):
        self.userMat = np.random.normal(0, 1.0, (self.userNum, self.dim))
        self.itemMat = np.random.normal(0, 1.0, (self.itemNum, self.dim))

        for _ in range(self.itera):
            for user_id in self.train_user_items.keys():
                item_list = self.train_user_items[user_id]
                neg_item_list = list(self.train_items - set(item_list))
                for item_id in item_list:
                    Pu = self.userMat[user_id]
                    Qi = self.itemMat[item_id]
                    neg_item_id = random.choice(neg_item_list)
                    Qk = self.itemMat[neg_item_id]
                    eik = np.dot(Pu, Qi) - np.dot(Pu, Qk)
                    logisticResult = logistic.cdf(-eik)
                    # calculate every gradient
                    gradient_pu = logisticResult * (Qk - Qi) + self.alpha * Pu
                    gradient_qi = logisticResult * (-Pu) + self.alpha * Qi
                    gradient_qk = logisticResult * (Pu) + self.alpha * Qk
                    # update every vector
                    self.userMat[user_id] = Pu - self.learning_rate * gradient_pu
                    self.itemMat[item_id] = Qi - self.learning_rate * gradient_qi
                    self.itemMat[neg_item_id] = Qk - self.learning_rate * gradient_qk

            P_50, MRR_50 = self.evalution()
            self.logger.info(self.data_type + ',' + 'P@50' + ' = ' + str(P_50))
            self.logger.info(self.data_type + ',' + 'MRR@50' + ' = ' + str(MRR_50) + '\n')


if __name__ == '__main__':
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    index = 5
    d = 50
    itera = 100
    model = BPR(type[index], d, itera)
    model.standard_BPR()
