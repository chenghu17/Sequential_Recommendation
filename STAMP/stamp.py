import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import logging.config


# 用户每天行为当作一个session，last item为预测
# 论文中是使用的对sequence进行preprocess，划分为[[s1],s2],[[s1,s2],s3],[[s1,s2,s3],s4]...

class data_generation():
    def __init__(self, type, neg_number):
        print('init')
        self.data_type = type
        self.train_dataset = './data/' + self.data_type + '/' + self.data_type + '_train_dataset.csv'
        self.test_dataset = './data/' + self.data_type + '/' + self.data_type + '_test_dataset.csv'

        self.train_sessions = []  # 当前的session
        self.train_items = []  # 每个session最后一个item
        self.train_neg_items = []  # 每个session对应的负样本

        self.neg_number = neg_number

        self.test_sessions = []
        self.test_real_items = []

        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0

    def gen_train_data(self):
        data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                is_first_line = 0
            else:
                sessions = [i for i in line[1].split('@')]
                for s in sessions:
                    neg_item_set = []
                    current_session = [int(it) for it in s.split(':')]
                    if len(current_session) < 2:
                        continue
                    self.train_items.append(current_session[-1])
                    self.train_sessions.append(current_session[:-1])
                    neg_item_set = self.gen_neg(current_session)
                    self.train_neg_items.append(neg_item_set)
                    self.records_number += 1

    def gen_test_data(self):
        data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')

        sub_index = self.shuffle(len(data.values))
        data = data.values[sub_index]

        for line in data:
            current_session = [int(i) for i in line[1].split(':')]
            if len(current_session) < 2:
                continue
            self.test_real_items.append(current_session[-1])
            self.test_sessions.append(current_session[:-1])

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index

    def gen_neg(self, current_session):
        count = 0
        neg_item_set = list()
        while count < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in current_session:
                neg_item_set.append(neg_item)
                count += 1
        return neg_item_set

    def gen_train_batch_data(self, batch_size):
        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0
        batch_item = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_session = self.train_sessions[self.train_batch_id]
        batch_neg_items = self.train_neg_items[self.train_batch_id]
        self.train_batch_id = self.train_batch_id + batch_size

        return batch_item, batch_session, batch_neg_items

    def gen_test_batch_data(self, batch_size):
        l = len(self.test_real_items)
        if self.test_batch_id == l:
            self.test_batch_id = 0
        batch_session = self.test_sessions[self.test_batch_id]
        self.test_batch_id = self.test_batch_id + batch_size

        return batch_session


class stamp():
    def __init__(self, data_type, neg_number, K, itera, global_dimension):
        print('init ... ')
        self.input_data_type = data_type

        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()

        self.dg = data_generation(self.input_data_type, neg_number)
        # 数据格式化
        self.dg.gen_train_data()
        self.dg.gen_test_data()

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number

        self.test_sessions = self.dg.test_sessions
        self.test_real_items = self.dg.test_real_items

        self.global_dimension = global_dimension
        self.batch_size = 1
        self.K = K
        self.results = []  # 可用来保存test每个用户的预测结果，最终计算precision

        self.step = 0
        self.iteration = itera

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=-np.sqrt(3 / self.global_dimension))

        self.real_item = tf.placeholder(tf.int32, shape=[None], name='item_id')

        self.neg_items = tf.placeholder(tf.int32, shape=[None], name='neg_items')

        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')

        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])

        self.w_0 = tf.get_variable('w_0', initializer=self.initializer_param,
                                   shape=[1, self.global_dimension])
        self.w_1 = tf.get_variable('w_1', initializer=self.initializer_param,
                                   shape=[self.global_dimension, self.global_dimension])
        self.w_2 = tf.get_variable('w_2', initializer=self.initializer_param,
                                   shape=[self.global_dimension, self.global_dimension])
        self.w_3 = tf.get_variable('w_3', initializer=self.initializer_param,
                                   shape=[self.global_dimension, self.global_dimension])
        self.b_a = tf.get_variable('b_attention', initializer=self.initializer_param,
                                   shape=[self.global_dimension, 1])

        self.h_w_a = tf.get_variable('h_w_a', initializer=self.initializer_param,
                                     shape=[self.global_dimension, self.global_dimension])
        self.h_w_t = tf.get_variable('h_w_t', initializer=self.initializer_param,
                                     shape=[self.global_dimension, self.global_dimension])
        self.h_b_a = tf.get_variable('h_b_a', initializer=self.initializer_param,
                                     shape=[1, self.global_dimension])
        self.h_b_t = tf.get_variable('h_b_t', initializer=self.initializer_param,
                                     shape=[1, self.global_dimension])

    def Attention_Net(self, current_session_embedding, memory_t):
        memory_s = tf.expand_dims(tf.reduce_mean(current_session_embedding, axis=0), axis=0)

        weight = tf.matmul(self.w_0, tf.nn.sigmoid(tf.add(
            tf.add(tf.matmul(self.w_1, tf.transpose(current_session_embedding)),
                   tf.matmul(self.w_2, tf.transpose(memory_t))),
            tf.add(tf.matmul(self.w_3, tf.transpose(memory_s)), self.b_a))))

        memory_a = tf.expand_dims(tf.reduce_sum(tf.multiply(current_session_embedding, tf.transpose(weight)), axis=0),
                                  axis=0)
        return memory_a

    def MLP_Cell_A(self, memory_a):
        # 1*d
        h_s = tf.tanh(tf.add(tf.matmul(memory_a, self.h_w_a), self.h_b_a))
        return h_s

    def MLP_Cell_B(self, memory_t):
        # 1*d
        h_t = tf.tanh(tf.add(tf.matmul(memory_t, self.h_w_t), self.h_b_t))
        return h_t

    def Trilinear_Composition(self, h_s, h_t, input_session_embedding):
        # self.item_embedding_matrix 即为候选item集合
        score = tf.nn.sigmoid(tf.matmul(h_s, tf.transpose(tf.multiply(input_session_embedding, h_t))))
        y_predict = tf.nn.softmax(score)
        return y_predict

    def loss_function(self, positive_result, neg_result):
        loss = (-1) * (tf.reduce_sum(tf.log(positive_result)) + tf.reduce_sum(tf.log(1 - neg_result)))
        return loss

    def build_model(self):
        print('building model ... ')

        real_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.real_item)
        neg_items_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_items)
        current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        # 取current_session_embedding最后一个
        memory_t = tf.expand_dims(current_session_embedding[-1], axis=0)
        # attention
        memory_a = self.Attention_Net(current_session_embedding, memory_t)
        # 经过MLP得到hidden state
        h_s = self.MLP_Cell_A(memory_a)
        h_t = self.MLP_Cell_B(memory_t)

        # calculate loss by cross entropy
        positive_result = self.Trilinear_Composition(h_s, h_t, real_item_embedding)
        neg_result = self.Trilinear_Composition(h_s, h_t, neg_items_embedding)
        self.intention_loss = self.loss_function(positive_result, neg_result)

        y_predict = self.Trilinear_Composition(h_s, h_t, self.item_embedding_matrix)
        self.top_value, self.top_index = tf.nn.top_k(y_predict, k=self.K, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            loss = 0
            for _ in range(self.iteration):
                print('new iteration begin ... ')
                print('iteration: ', str(iter))

                while self.step * self.batch_size < self.dg.records_number:
                    batch_item, batch_session, batch_neg_items = self.dg.gen_train_batch_data(self.batch_size)
                    _, loss = self.sess.run([intention_optimizer, self.intention_loss],
                                            feed_dict={self.real_item: batch_item,
                                                       self.current_session: batch_session,
                                                       self.neg_items: batch_neg_items})
                    self.step += 1
                print('loss = ', loss)
                print('eval ...')
                self.evolution()
                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

    def precision_k(self, pre_top_k, true_items):
        right_pre = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i]:
                right_pre += 1
        return right_pre / (user_number * self.K)

    def recall_k(self, pre_top_k, true_items):
        right_pre = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i]:
                right_pre += 1
        return right_pre / user_number

    def MRR_k(self, pre_top_k, true_items):
        MRR_rate = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i]:
                index = pre_top_k[i].tolist()[0].index(true_items[i])
                MRR_rate += 1 / (index + 1)
        return MRR_rate / user_number

    def evolution(self):
        pre_top_k = []

        for _ in self.test_real_items:
            batch_session = self.dg.gen_test_batch_data(self.batch_size)
            top_k_value, top_index = self.sess.run([self.top_value, self.top_index],
                                                   feed_dict={self.current_session: batch_session})
            pre_top_k.append(top_index)

        P = self.recall_k(pre_top_k, self.test_real_items)
        MRR = self.MRR_k(pre_top_k, self.test_real_items)

        self.logger.info(self.input_data_type + ',' + 'recall@' + str(self.K) + ' = ' + str(P))
        self.logger.info(self.input_data_type + ',' + 'MRR@' + str(self.K) + ' = ' + str(MRR))

        return


if __name__ == '__main__':
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens']
    neg_number = 10
    K = 20
    itera = 200
    global_dimension = 20
    index = 0
    model = stamp(type[index], neg_number, K, itera, global_dimension)
    model.build_model()
    model.run()
