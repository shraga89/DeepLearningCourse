import numpy as np
import pandas as pd
from sympy.functions.special.polynomials import gegenbauer

import statistics
from datetime import datetime
from random import randint

import utils

class NiceData():

    def __init__(self,data_path,feature_mapping,channel_mapping):
        channels = pd.read_csv(channel_mapping,header=0)
        self.genres = {}
        for i in channels.iterrows():
            # print i[1][1]
            self.genres[i[1][1]] = int(i[1][0])
        print(self.genres)
        self.feature_map = pd.read_csv(feature_mapping)
        self.data = self.create_data_frame(data_path)
        self.channel_map  = channel_mapping
        self.adjust_data()

    def create_data_frame(self,data_path):
        col = self.feature_map.name
        print col
        # cols_to_use =[i for i in range(50) if i not in [2,7,21,22,23,24,25,26,27,29]]
        # cols_to_use.append(95)
        self.df = pd.read_csv(data_path,names=col, header=0)

    def adjust_data(self):
        for i in self.df.columns.values:
            print i
            if i not in ['DEVICE_ID','EVENT_TIME','PROGRAM_GENRE','HOUSEHOLD_ID']:
                temp = [self.df, pd.get_dummies(self.df[i],prefix=[i])]
                self.df = pd.concat(temp, axis=1)
                self.df = self.df.drop([i], axis=1)
        # for i in ['EVENT_TIME', 'AIR_DATE', 'AIR_TIME']:
        #     print(i)
        #     temp = pd.to_datetime(self'.df[i])
        #     temp = (temp - temp.min()) / np.timedelta64(1, 'D')
        #     self.df = pd.concat([self.df, temp], axis=1)
        #     self.df = self.df.drop([i], axis=1)

    def create_sequences(self):
        sequences = self.df.groupby(['DEVICE_ID'])
        sequences = sequences.groups
        train_data={}
        target={}
        for p, key in enumerate(sequences):
            print "sequence " + str(p)
            #if p<69369:
                #continue
            if p==5000:
                break
            data_point = None
            seq_target=[]
            sorted_indexs = self.df.ix[sequences[key].values].sort('EVENT_TIME').index.values
            # city = randint(1,45)
            for idx in sorted_indexs:
                # print self.df.ix[idx]['PROGRAM_GENRE']
                genre = self.genres[self.df.ix[idx]['PROGRAM_GENRE']]
                # print genre
                seq_target.append(genre)
                temp = self.df.ix[idx].drop(['EVENT_TIME','PROGRAM_GENRE','HOUSEHOLD_ID'])
                # temp = self.df.ix[idx]
                if data_point is None:
                    data_point = temp.as_matrix()
                else:
                    data_point = np.concatenate((data_point,temp))
            # data.df = data.df.drop(sorted_indexs)
            train_data[key]=list(data_point)
            target[key] = seq_target
            print len(train_data[key])
        utils.write_to_json('trainA.json',train_data)
        utils.write_to_json('trainA_target.json',target)

    def statistics(self):
        statistics.sequences_length_stat(self.df)
        statistics.platform_usage(self.channels, self.df)
        statistics.channel_movements(self.channels, self.df)

if __name__ == '__main__':
    #statistics.platform_usage('DataDesc.csv')
    #statistics.sequences_length_stat()
    data = NiceData('final_data_set/data_set.csv','final_data_set/headers.csv','final_data_set/genres.csv')
    data.create_sequences()
