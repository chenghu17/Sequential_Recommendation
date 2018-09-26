import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import logging.config


class data_generation():
    def __init__(self, type, neg_number):
        print('init')
        self.data_type = type
        self.dataset = './data/' + self.data_type + '/' + self.data_type + '_train_test_dataset.csv'

        self.train_users = []
        self.train_sessions = []  # 当前的session
        self.train_last_sessions = []  # 上一个session
        self.train_neg_items = []  # 每个session对应的负样本

        self.neg_number = neg_number

        self.test_users = []
        self.test_sessions = []
        self.test_last_sessions = []

        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0

    def gen_train_test_data(self):
        data = pd.read_csv(self.dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        for line in data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                is_first_line = 0
            else:
                sessions = [i for i in line[1].split('@')]
                user_id = int(line[0])
                for i in range(len(sessions) - 1):
                    last_session = [int(it) for it in sessions[i].split(':')]
                    current_session = [int(it) for it in sessions[i + 1].split(':')]
                    if i == len(sessions) - 2:
                        self.test_users.append(user_id)
                        self.test_sessions.append(current_session)
                        self.test_last_sessions.append(last_session)
                    else:
                        self.train_users.append(user_id)
                        neg_item_set = []
                        self.train_last_sessions.append(last_session)
                        self.train_sessions.append(current_session)
                        neg_item_set = self.gen_neg(current_session)
                        self.train_neg_items.append(neg_item_set)
                        self.records_number += 1

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
        batch_user_id = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        batch_last_session = self.train_last_sessions[self.train_batch_id]
        batch_session = self.train_sessions[self.train_batch_id]
        batch_neg_items = self.train_neg_items[self.train_batch_id]
        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user_id, batch_last_session, batch_session, batch_neg_items

    def gen_test_batch_data(self, batch_size):
        l = len(self.test_users)
        if self.test_batch_id == l:
            self.test_batch_id = 0
        bacth_user_id = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
        batch_last_session = self.test_last_sessions[self.test_batch_id]
        self.test_batch_id = self.test_batch_id + batch_size

        return bacth_user_id, batch_last_session


class stamp():
    def __init__(self, data_type, neg_number):
        print('init ... ')
        self.input_data_type = data_type

        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()

        self.dg = data_generation(self.input_data_type, neg_number)
        # 数据格式化
        self.dg.gen_train_test_data()

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number

        self.test_users = self.dg.test_users
        self.test_sessions = self.dg.test_sessions

        self.global_dimension = 20
        self.batch_size = 1
        self.K = 20
        self.results = []  # 可用来保存test每个用户的预测结果，最终计算precision

        self.step = 0
        self.iteration = 100
        self.lamada_u_v = 0.001

        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=-np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.last_session = tf.placeholder(tf.int32, shape=[None], name='last_session')
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.neg_items = tf.placeholder(tf.int32, shape=[None], name='neg_items')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])

    def avg_pooling(self, input):
        output = tf.expand_dims(tf.reduce_mean(input, axis=0), axis=0)
        return output

    def max_pooling(self, input):
        output = tf.expand_dims(tf.reduce_max(input, axis=0), axis=0)
        return output

    def hybrid_user_embedding(self, user_embedding, last_session_embedding):
        f1 = self.avg_pooling(last_session_embedding)
        f2 = self.avg_pooling(tf.concat([user_embedding, f1], 0))
        return f2

    def predict(self, hybrid_user_embedding, input_session_embedding):
        p = tf.transpose(tf.matmul(input_session_embedding, tf.transpose(hybrid_user_embedding)))
        return p

    def loss_function(self, positive_result, negtive_result, user_embedding, current_session_embedding,
                      last_session_embedding, neg_items_embedding):
        loss = (-1) * (tf.reduce_sum(tf.log(tf.nn.sigmoid(positive_result))) + tf.reduce_max(
            tf.log(tf.nn.sigmoid((-1) * negtive_result)))) + self.lamada_u_v * tf.nn.l2_loss(
            user_embedding) + self.lamada_u_v * tf.nn.l2_loss(
            current_session_embedding) + self.lamada_u_v * tf.nn.l2_loss(
            last_session_embedding) + self.lamada_u_v * tf.nn.l2_loss(neg_items_embedding)

        return loss

    def build_model(self):
        print('building model ... ')

        user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        last_session_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.last_session)
        current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        neg_items_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.last_session)

        hybrid = self.hybrid_user_embedding(user_embedding, last_session_embedding)

        positive_result = self.predict(hybrid, current_session_embedding)
        negtive_result = self.predict(hybrid, neg_items_embedding)

        # calculate loss by cross entropy
        self.intention_loss = self.loss_function(positive_result, negtive_result, user_embedding,
                                                 current_session_embedding,
                                                 last_session_embedding, neg_items_embedding)

        y_predict = self.predict(hybrid, self.item_embedding_matrix)

        self.top_value, self.top_index = tf.nn.top_k(y_predict, k=self.K, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            for _ in range(self.iteration):
                print('new iteration begin ... ')
                print('iteration: ', str(iter))

                while self.step * self.batch_size < self.dg.records_number:
                    batch_user_id, batch_last_session, batch_session, batch_neg_items = self.dg.gen_train_batch_data(
                        self.batch_size)
                    self.sess.run(intention_optimizer,
                                  feed_dict={self.user_id: batch_user_id,
                                             self.last_session: batch_last_session,
                                             self.current_session: batch_session,
                                             self.neg_items: batch_neg_items})
                    self.step += 1
                print('eval ...')
                self.evolution()
                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

            # 保存模型
            # self.save()

    # def save(self):
    #     item_latent_factors, the_first_w, the_second_w, the_first_bias, the_second_bias = self.sess.run(
    #         [self.item_embedding_matrix, self.the_first_w, self.the_second_w,
    #          self.the_first_bias, self.the_second_bias])
    #
    #     t = pd.DataFrame(item_latent_factors)
    #     t.to_csv('./model_result/gowalla/item_latent_factors')
    #
    #     t = pd.DataFrame(the_first_w)
    #     t.to_csv('./model_result/gowalla/the_first_w')
    #
    #     t = pd.DataFrame(the_second_w)
    #     t.to_csv('./model_result/gowalla/the_second_w')
    #
    #     t = pd.DataFrame(the_first_bias)
    #     t.to_csv('./model_result/gowalla/the_first_bias')
    #
    #     t = pd.DataFrame(the_second_bias)
    #     t.to_csv('./model_result/gowalla/the_second_bias')

    # return

    def precision_recall_k(self, pre_top_k, test_sessions):
        right_pre = 0
        right_recall = 0
        line = len(pre_top_k)
        for i in range(line):
            # 取list的交集
            # pre_top_k[i]为ndarray类型，需要.tolist()转换为list，而得到的是[[x,x,x,x]]，所以取[0]
            correct = len(set(pre_top_k[i].tolist()[0]).intersection(set(test_sessions[i])))
            right_pre += correct / len(pre_top_k[i])
            right_recall += correct / len(set(test_sessions[i]))
        return right_pre / line, right_recall / line

    def evolution(self):
        pre_top_k = []

        for _ in self.test_users:
            bacth_user_id, batch_last_session = self.dg.gen_test_batch_data(self.batch_size)
            top_k_value, top_index = self.sess.run([self.top_value, self.top_index],
                                                   feed_dict={self.user_id: bacth_user_id,
                                                              self.last_session: batch_last_session})
            pre_top_k.append(top_index)

        precision, recall = self.precision_recall_k(pre_top_k, self.test_sessions)
        self.logger.info('precision@' + str(self.K) + ' = ' + str(precision))
        self.logger.info('precision@' + str(self.K) + ' = ' + str(recall))
        self.logger.info('precision@' + str(self.K) + ' = ' + str(2 * precision * recall / (precision + recall)))

        return


if __name__ == '__main__':
    type = ['tallM', 'gowalla']
    neg_number = 10
    model = stamp(type[0], neg_number)
    model.build_model()
    model.run()
