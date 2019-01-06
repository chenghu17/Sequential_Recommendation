# author：hucheng

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import random


class data_generation():
    def __init__(self, type, neg_number):
        print('init')
        self.data_type = type
        self.train_dataset = './data/' + self.data_type + '/' + self.data_type + '_train_dataset.csv'
        self.test_dataset = './data/' + self.data_type + '/' + self.data_type + '_test_dataset.csv'

        self.train_users = []
        self.train_sessions = []  # 当前的session
        self.train_items = []
        self.train_neg_items = []  # 每个session对应的负样本

        self.neg_number = neg_number

        self.test_users = []
        self.test_sessions = []
        self.test_items = []

        self.user_number = 0
        self.item_number = 0

    def gen_train_data(self):
        train_data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in train_data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                is_first_line = 0
            else:
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                for i in range(len(sessions)):
                    cur_session = [int(it) for it in sessions[i].split(':')]
                    item = random.choice(cur_session)
                    cur_session.remove(item)
                    neg_items_list = self.gen_neg(cur_session)
                    self.train_users.append(user_id)
                    self.train_sessions.append(cur_session)
                    self.train_items.append(item)
                    self.train_neg_items.append(neg_items_list)

    def gen_neg(self, current_session):
        neg_item_set = set()
        while len(neg_item_set) < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in current_session:
                neg_item_set.add(neg_item)
        return list(neg_item_set)

    def gen_test_data(self):
        test_data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        # 对于ndarray进行sample得到test目标数据
        sub_index = self.shuffle(len(test_data.values))
        data = test_data.values[sub_index]
        for line in data:
            user_id = int(line[0])
            if user_id in self.train_users:
                current_session = [int(i) for i in line[1].split(':')]
                if len(current_session) < 2:
                    continue
                item = random.choice(current_session)
                current_session.remove(item)
                self.test_items.append(int(item))
                self.test_sessions.append(current_session)
                self.test_users.append(user_id)

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index


class hrm():
    def __init__(self, data_type, neg_number, K, itera, global_dimension):

        print('init ... ')
        self.input_data_type = data_type

        self.dg = data_generation(self.input_data_type, neg_number)
        # 数据格式化
        self.dg.gen_train_data()
        self.dg.gen_test_data()

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number

        self.train_users = self.dg.train_users
        self.train_items = self.dg.train_items
        self.train_sessions = self.dg.train_sessions
        self.train_neg_items = self.dg.train_neg_items

        self.test_users = self.dg.test_users
        self.test_items = self.dg.test_items
        self.test_sessions = self.dg.test_sessions

        self.global_dimension = global_dimension
        self.K = K

        # 日志基本配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        fh = logging.FileHandler('hrm_' + self.input_data_type + '_d_' + str(self.global_dimension), mode='a',
                                 encoding=None,
                                 delay=False)
        self.logger.addHandler(fh)

        self.iteration = itera
        self.lamada_u_v = 0.01

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.neg_items = tf.placeholder(tf.int32, shape=[None], name='neg_items')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])

    def avg_pooling(self, input):
        output = tf.expand_dims(tf.reduce_mean(input, axis=0), axis=0)
        return output

    def max_pooling(self, input):
        output = tf.expand_dims(tf.reduce_max(input, axis=0), axis=0)
        return output

    def hybrid_user_embedding(self, user_embedding, cur_session_embedding):
        f1 = self.max_pooling(cur_session_embedding)
        f2 = self.max_pooling(tf.concat([user_embedding, f1], 0))
        return f2

    def predict(self, hybrid_user_embedding, input_session_embedding):
        p = tf.transpose(tf.matmul(input_session_embedding, tf.transpose(hybrid_user_embedding)))
        return p

    def loss_function(self, positive_result, negative_result, hybrid,
                      item_embedding, neg_items_embedding):
        loss = (-1) * (tf.reduce_sum(tf.log(tf.nn.sigmoid(positive_result))) + tf.reduce_sum(
            tf.log(tf.nn.sigmoid((-1) * negative_result)))) + self.lamada_u_v * tf.nn.l2_loss(
            hybrid) + self.lamada_u_v * tf.nn.l2_loss(
            item_embedding) + self.lamada_u_v * tf.nn.l2_loss(neg_items_embedding)

        return loss

    def build_model(self):
        print('building model ... ')

        user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        neg_items_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_items)

        hybrid = self.hybrid_user_embedding(user_embedding, current_session_embedding)

        positive_result = self.predict(hybrid, item_embedding)
        negtive_result = self.predict(hybrid, neg_items_embedding)

        # calculate loss by cross entropy
        self.intention_loss = self.loss_function(positive_result, negtive_result, hybrid,
                                                 item_embedding, neg_items_embedding)

        y_predict = self.predict(hybrid, self.item_embedding_matrix)

        self.top_value, self.top_index = tf.nn.top_k(y_predict, k=self.K, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            for iter in range(self.iteration):
                print('new iteration begin ... ')
                print('iteration: ', iter)
                all_loss = 0
                for inde in range(len(self.train_users)):
                    cur_session = self.train_sessions[inde]
                    cur_user_id = self.train_users[inde]
                    cur_item_id = self.train_items[inde]
                    cur_neg_items = self.train_neg_items[inde]

                    batch_user_id = list()
                    batch_user_id.append(cur_user_id)
                    batch_item_id = list()
                    batch_item_id.append(cur_item_id)

                    _, loss = self.sess.run([intention_optimizer, self.intention_loss],
                                            feed_dict={self.user_id: batch_user_id,
                                                       self.item_id: batch_item_id,
                                                       self.current_session: cur_session,
                                                       self.neg_items: cur_neg_items})
                    all_loss += loss
                print('all_loss', all_loss)
                print('eval ...')
                self.evolution()

    def P_k(self, pre_top_k, true_items):
        right_pre = 0
        record_number = len(true_items)
        for i in range(record_number):
            if true_items[i] in pre_top_k[i][0]:
                right_pre += 1
        return right_pre / record_number

    def MRR_k(self, pre_top_k, true_items):
        MRR_rate = 0
        record_number = len(true_items)
        for i in range(record_number):
            if true_items[i] in pre_top_k[i][0]:
                index = pre_top_k[i].tolist()[0].index(true_items[i])
                MRR_rate += 1 / (index + 1)
        return MRR_rate / record_number

    def evolution(self):
        pre_top_k = []

        for inde in range(len(self.test_users)):
            cur_session = self.test_sessions[inde]
            cur_user_id = self.test_users[inde]
            cur_item_id = self.test_items[inde]

            batch_user_id = list()
            batch_user_id.append(cur_user_id)
            batch_item_id = list()
            batch_item_id.append(cur_item_id)

            # 此处需要改成当前item与user，还有当前session，以及目标item

            top_k_value, top_index = self.sess.run([self.top_value, self.top_index],
                                                   feed_dict={self.user_id: batch_user_id,
                                                              self.current_session: cur_session})
            pre_top_k.append(top_index)

        P_50 = self.P_k(pre_top_k, self.test_items)
        MRR_50 = self.MRR_k(pre_top_k, self.test_items)

        self.logger.info('P@' + str(self.K) + ' = ' + str(P_50))
        self.logger.info('MRR@' + str(self.K) + ' = ' + str(MRR_50))

        return


if __name__ == '__main__':
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    neg_number = 10
    K = 50
    itera = 200
    global_dimension = 50
    index = 5
    model = hrm(type[index], neg_number, K, itera, global_dimension)
    model.build_model()
    model.run()
