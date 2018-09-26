import pandas as pd
import numpy as np
import os


# 用于生成SHAN中训练的数据格式，即userid,id:id:id@id:id:id...

class generate(object):

    def __init__(self, rootPath, dataPath, sessPath):
        self._data = pd.read_csv(dataPath)
        self.rootPath = rootPath
        self.sessPath = sessPath

    def stati_data(self):
        print('总数据量:', len(self._data))
        print('总session数:', len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('平均session长度:', len(self._data) / len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('总user数:', len(self._data.drop_duplicates('use_ID')))
        print('平均每个用户拥有的session个数:',
              len(self._data.drop_duplicates(['use_ID', 'time'])) / len(self._data.drop_duplicates('use_ID')))
        print('总item数:', len(self._data.drop_duplicates('ite_ID')))
        print('数据集时间跨度：', min(self._data.time), '~', max(self._data.time))

    def reform_u_i_id(self):
        # 将数据中的item和user重新编号，然后再生成session
        user_to_id = {}
        item_to_id = {}
        # 对user进行重新编号
        user_count = 0
        item_count = 0
        for i in range(len(self._data)):
            # 对user 和 item同时进行重新编号
            u_id = self._data.at[i, 'use_ID']
            i_id = self._data.at[i, 'ite_ID']
            if u_id in user_to_id.keys():
                self._data.at[i, 'use_ID'] = user_to_id[u_id]
            else:
                user_to_id[u_id] = user_count
                self._data.at[i, 'use_ID'] = user_count
                user_count += 1
            if i_id in item_to_id.keys():
                self._data.at[i, 'ite_ID'] = item_to_id[i_id]
            else:
                item_to_id[i_id] = item_count
                self._data.at[i, 'ite_ID'] = item_count
                item_count += 1
        self._data.to_csv(self.rootPath + 'middle_data.csv', index=False)
        print('user_count', user_count)
        print('item_count', item_count)

    # 按照实验设计，test的session是从数据集的最后一个月随机抽取百分之二十的session得到的
    # TallM中使用的test集合是每个用户最后一个session
    def generate_train_test_session(self):
        print('statistics ... ')
        self.stati_data()  # 统计数据集
        print('encode ... ')
        self.reform_u_i_id()  # 重新编码user和item
        print('generate train and test session ... ')
        self._data = pd.read_csv(self.rootPath + 'middle_data.csv')

        os.remove(self.rootPath + 'middle_data.csv')
        session_train_test_path = self.sessPath + '_train_test_dataset.csv'
        if os.path.exists(session_train_test_path):
            os.remove(session_train_test_path)

        # for train session
        # 要考虑最后一个session，目前循环中没有考虑最后一个session
        with open(session_train_test_path, 'a') as session_train_file:
            user_num = len(self._data['use_ID'].drop_duplicates())
            # users = len(self._train_data['use_ID'].drop_duplicates())
            item_num = len(self._data['ite_ID'].drop_duplicates())
            session_train_file.write(str(user_num) + ',' + str(item_num) + '\n')
            last_userid = self._data.at[0, 'use_ID']
            last_time = self._data.at[0, 'time']
            session = str(last_userid) + ',' + str(self._data.at[0, 'ite_ID'])
            for i in range(1, len(self._data)):
                # 文件使用降序打开
                # 最终session的格式为user_id,item_id:item_id...@item_id:item_id...@...
                userid = self._data.at[i, 'use_ID']
                itemid = self._data.at[i, 'ite_ID']
                time = self._data.at[i, 'time']
                if userid == last_userid and time == last_time:
                    # 需要将session写入到文件中，然后开始
                    session += ":" + str(itemid)
                elif userid != last_userid:
                    session_train_file.write(session + '\n')
                    last_userid = userid
                    last_time = time
                    session = str(userid) + ',' + str(itemid)
                else:
                    session += '@' + str(itemid)
                    last_time = time
            session_train_file.write(session + '\n')


if __name__ == '__main__':
    datatype = ['tallM', 'gowalla']
    i = 1
    rootPath = datatype[i] + '/'
    dataPath = datatype[i] + '/' + datatype[i] + '_data.csv'
    sessPath = datatype[i] + '/' + datatype[i]
    object = generate(rootPath, dataPath, sessPath)
    # object.stati_data()
    object.generate_train_test_session()
