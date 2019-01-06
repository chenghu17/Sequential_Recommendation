# author：hucheng

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
import logging
import logging.config


class data_generation():
    def __init__(self, type, neg_number):
        print('init')
        self.data_type = type
        self.train_dataset = './data/' + self.data_type + '/' + self.data_type + '_train_dataset.csv'
        self.test_dataset = './data/' + self.data_type + '/' + self.data_type + '_test_dataset.csv'


        self.train_users = []
        self.train_sessions = []  # 当前的session
        self.train_items = []  # 随机采样得到的positive
        self.train_neg_items = []  # 随机采样得到的negative
        self.train_pre_sessions = []  # 之前的session集合

        self.test_users = []
        self.test_candidate_items = []
        self.test_sessions = []
        self.test_pre_sessions = []
        self.test_real_items = []

        self.neg_number = neg_number
        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0

    def gen_train_data(self):
        self.data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in self.data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                self.user_purchased_item = dict()  # 保存每个用户购买记录，可用于train时负采样和test时剔除已打分商品
                is_first_line = 0
            else:
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                if size < 2:
                    continue
                the_first_session = [int(i) for i in sessions[0].split(':')]
                self.train_pre_sessions.append(the_first_session)
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item[user_id] = tmp
                for j in range(1, size):
                    # 每个用户的每个session在train_users中都对应着其user_id，不一定是连续的
                    self.train_users.append(user_id)
                    # test = sessions[j].split(':')
                    current_session = [int(it) for it in sessions[j].split(':')]
                    neg = self.gen_neg(user_id)
                    self.train_neg_items.append(neg)
                    # 将当前session加入到用户购买的记录当中
                    # 之所以放在这个位置，是因为在选择测试item时，需要将session中的一个item移除、
                    # 如果放在后面操作，当前session中其实是少了一个用来做当前session进行预测的item
                    if j != 1:
                        tmp = copy.deepcopy(self.user_purchased_item[user_id])
                        self.train_pre_sessions.append(tmp)
                    tmp = copy.deepcopy(current_session)
                    self.user_purchased_item[user_id].extend(tmp)
                    # 随机挑选一个作为prediction item
                    item = random.choice(current_session)
                    self.train_items.append(item)
                    current_session.remove(item)
                    self.train_sessions.append(current_session)
                    self.records_number += 1

    def gen_test_data(self):
        self.data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        self.test_candidate_items = list(range(self.item_number))

        # 对于ndarray进行sample得到test目标数据
        sub_index = self.shuffle(len(self.data.values))
        data = self.data.values[sub_index]

        for line in data:
            user_id = int(line[0])
            if user_id in self.user_purchased_item.keys():
                current_session = [int(i) for i in line[1].split(':')]
                if len(current_session) < 2:
                    continue
                self.test_users.append(user_id)
                item = random.choice(current_session)
                self.test_real_items.append(int(item))
                current_session.remove(item)
                self.test_sessions.append(current_session)
                self.test_pre_sessions.append(self.user_purchased_item[user_id])

        # batch_user = self.test_users[user_id:user_id + batch_size]
        # batch_item = self.test_candidate_items
        # batch_session = self.test_sessions[user_id]
        # batch_pre_session = self.test_pre_sessions[user_id]

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index

    def gen_neg(self, user_id):
        count = 0
        neg_item_set = list()
        while count < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in self.user_purchased_item[user_id]:
                neg_item_set.append(neg_item)
                count += 1
        return neg_item_set

    def gen_train_batch_data(self, batch_size):
        # l = len(self.train_users)

        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0

        batch_user = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        batch_item = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_session = self.train_sessions[self.train_batch_id]
        # batch_neg_item = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_neg_item = self.train_neg_items[self.train_batch_id]
        batch_pre_session = self.train_pre_sessions[self.train_batch_id]

        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session

    def gen_test_batch_data(self, batch_size):
        l = len(self.test_users)

        if self.test_batch_id == l:
            self.test_batch_id = 0

        batch_user = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
        batch_item = self.test_candidate_items
        batch_session = self.test_sessions[self.test_batch_id]
        batch_pre_session = self.test_pre_sessions[self.test_batch_id]

        self.test_batch_id = self.test_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_pre_session


class shan():
    # data_type :  TallM / GWL
    def __init__(self, data_type, neg_number, itera, global_dimension):
        print('init ... ')
        self.input_data_type = data_type

        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()
        fh = logging.FileHandler('shan_log_' + data_type + '_d_' + str(global_dimension), mode='a', encoding=None,
                                 delay=False)
        self.logger.addHandler(fh)

        self.dg = data_generation(self.input_data_type, neg_number)
        # 数据格式化
        self.dg.gen_train_data()
        self.dg.gen_test_data()

        self.train_user_purchased_item_dict = self.dg.user_purchased_item

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number
        self.neg_number = self.dg.neg_number

        self.test_users = self.dg.test_users
        self.test_candidate_items = self.dg.test_candidate_items
        self.test_sessions = self.dg.test_sessions
        self.test_pre_sessions = self.dg.test_pre_sessions
        self.test_real_items = self.dg.test_real_items

        self.global_dimension = global_dimension
        self.batch_size = 1
        self.results = []  # 可用来保存test每个用户的预测结果，最终计算precision

        self.step = 0
        self.iteration = itera
        self.lamada_u_v = 0.01
        self.lamada_a = 0.01

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        # 不管是当前的session，还是之前的session集合，在数据处理阶段都是一个数组，数组内容为item的编号
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.pre_sessions = tf.placeholder(tf.int32, shape=[None], name='pre_sessions')
        self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        self.the_first_w = tf.get_variable('the_first_w', initializer=self.initializer_param,
                                           shape=[self.global_dimension, self.global_dimension])
        self.the_second_w = tf.get_variable('the_second_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])
        self.the_first_bias = tf.get_variable('the_first_bias', initializer=self.initializer_param,
                                              shape=[self.global_dimension])
        self.the_second_bias = tf.get_variable('the_second_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])

    def attention_level_one(self, user_embedding, pre_sessions_embedding, the_first_w, the_first_bias):

        # 由于维度的原因，matmul和multiply方法要维度的变化
        # 最终weight为 1*n 的矩阵
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)), tf.transpose(user_embedding))))

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding, the_second_w,
                            the_second_bias):
        # 需要将long_user_embedding加入到current_session_embedding中来进行attention，
        # 论文中规定，long_user_embedding的表示也不会根据softmax计算得到的参数而变化。

        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                          the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def build_model(self):
        print('building model ... ')
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)

        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.pre_sessions_embedding,
                                                            self.the_first_w, self.the_first_bias)

        self.hybrid_user_embedding = self.attention_level_two(self.user_embedding, self.long_user_embedding,
                                                              self.current_session_embedding,
                                                              self.the_second_w, self.the_second_bias)

        # compute preference
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding))
        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding))
        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        self.regular_loss_u_v = tf.add(tf.add(self.lamada_u_v * tf.nn.l2_loss(self.user_embedding),
                                              self.lamada_u_v * tf.nn.l2_loss(self.item_embedding)),
                                       self.lamada_u_v * tf.nn.l2_loss(self.neg_item_embedding))
        self.regular_loss_a = tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w),
                                     self.lamada_a * tf.nn.l2_loss(self.the_second_w))
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)

        # 增加test操作，由于每个用户pre_sessions和current_session的长度不一样，
        # 所以无法使用同一个矩阵进行表示同时计算，因此每个user计算一次，将结果保留并进行统计
        # 注意，test集合的整个item_embeeding得到的是 [M*K]的矩阵，M为所有item的个数，K为维度
        self.top_value_10, self.top_index_10 = tf.nn.top_k(self.positive_element_wise, k=10, sorted=True)
        self.top_value_20, self.top_index_20 = tf.nn.top_k(self.positive_element_wise, k=20, sorted=True)
        self.top_value_50, self.top_index_50 = tf.nn.top_k(self.positive_element_wise, k=50, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            for iter in range(self.iteration):
                print('new iteration begin ... ')
                self.logger.info('iteration: '+str(iter))

                all_loss = 0
                while self.step * self.batch_size < self.dg.records_number:
                    # 按批次读取数据
                    batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions = self.dg.gen_train_batch_data(
                        self.batch_size)

                    _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                            feed_dict={self.user_id: batch_user,
                                                       self.item_id: batch_item,
                                                       self.current_session: batch_session,
                                                       self.neg_item_id: batch_neg_item,
                                                       self.pre_sessions: batch_pre_sessions
                                                       })
                    all_loss += loss
                    self.step += 1
                    # if self.step * self.batch_size % 5000 == 0:
                self.logger.info('loss = '+str(all_loss)+'\n')
                self.logger.info('eval ...')
                self.evolution()
                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

    def P_k(self, pre_top_k, true_items):
        right_pre = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                right_pre += 1
        return right_pre / user_number

    def MRR_k(self, pre_top_k, true_items):
        MRR_rate = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                index = pre_top_k[i].tolist()[0].index(true_items[i])
                MRR_rate += 1 / (index + 1)
        return MRR_rate / user_number

    def evolution(self):
        pre_top_k_10 = []
        pre_top_k_20 = []
        pre_top_k_50 = []

        for _ in self.test_users:
            batch_user, batch_item, batch_session, batch_pre_session = self.dg.gen_test_batch_data(
                self.batch_size)
            top_index_10, top_index_20, top_index_50 = self.sess.run(
                [self.top_index_10, self.top_index_20, self.top_index_50],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.current_session: batch_session,
                           self.pre_sessions: batch_pre_session})
            pre_top_k_10.append(top_index_10)
            pre_top_k_20.append(top_index_20)
            pre_top_k_50.append(top_index_50)

        P_10 = self.P_k(pre_top_k_10, self.test_real_items)
        MRR_10 = self.MRR_k(pre_top_k_10, self.test_real_items)

        P_20 = self.P_k(pre_top_k_20, self.test_real_items)
        MRR_20 = self.MRR_k(pre_top_k_20, self.test_real_items)

        P_50 = self.P_k(pre_top_k_50, self.test_real_items)
        MRR_50 = self.MRR_k(pre_top_k_50, self.test_real_items)

        self.logger.info(self.input_data_type + ',' + 'P@10' + ' = ' + str(P_10))
        self.logger.info(self.input_data_type + ',' + 'MRR@10' + ' = ' + str(MRR_10) + '\n')

        self.logger.info(self.input_data_type + ',' + 'P@20' + ' = ' + str(P_20))
        self.logger.info(self.input_data_type + ',' + 'MRR@20' + ' = ' + str(MRR_20) + '\n')

        self.logger.info(self.input_data_type + ',' + 'P@50' + ' = ' + str(P_50))
        self.logger.info(self.input_data_type + ',' + 'MRR@50' + ' = ' + str(MRR_50) + '\n')

        return


if __name__ == '__main__':
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    neg_number = 10
    itera = 100
    global_dimension = 50
    index = 5
    model = shan(type[index], neg_number, itera, global_dimension)
    model.build_model()
    model.run()
