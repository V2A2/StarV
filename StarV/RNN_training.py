"""
Recurrent Layer Class
Qing Liu, 12/05/2025
"""
from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class RNN_training(object):

    def __init__(self):
        pass

    def load_data(self):

        ''' Load Data '''

        directory = os.path.dirname(os.path.abspath(__file__))
        print("current directory:",directory)
        data_path = directory +"/util/data/CMAPSS/CMAPSSData"
        print("current data path:",data_path)

        index_names = ['unit_number', 'time_cycles']
        operational_names = ['OP_1', 'OP_2', 'OP_3']
        sensor_names = ['var_{}'.format(i+1) for i in range(0,21)]
        col_names = index_names + operational_names + sensor_names

        df_train = pd.read_csv(data_path + '/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
        df_test = pd.read_csv(data_path +'/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
        y_test = pd.read_csv(data_path +'/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])
        df_test.shape

        print("all_train_data_shape:",df_train.shape)
        print("all_test_data_shape:",df_test.shape)
        print("all_test_URL_shape:",y_test.shape)

        # train_samples = df_train.head(10)
        # pd.set_option('display.max_column', 30)
        # print("train_data_samples:",train_samples)/

        return df_train,df_test,y_test


    def Merged_with_RUL(self, df_data):

        ''' Add additinal column for currecnt RUL for each engine cycle'''

        total_cycles_per_engine = (df_data.groupby("unit_number")["time_cycles"].max())
        cycles_per_engine = total_cycles_per_engine.reset_index().rename(columns={"time_cycles": "total_cycles"})
        print("Number of cycles for each engine:")
        print(cycles_per_engine)

        df_merged_data = df_data.merge(cycles_per_engine, on="unit_number", how="left")

        df_merged_data["RUL"] = df_merged_data["total_cycles"] - df_merged_data["time_cycles"]
        
        # df_merged_data = df_merged_data.drop("total_cycles", axis=1) 

        print("\n Merged data with RUL:")
        print(df_merged_data)

        return df_merged_data


    def create_input_sequnces(self,engine_data, window_size):
       
        """
        Input:
        engine_data: dataframe with columns:
            unit_number, time_cycles, OP_1, OP_2, OP_3,
            var_1 ... var_21, total_cycles, RUL
        window_size: sliding window length ( time steps)

        Output:
            X: input sequences for all engines
            y: target RUL value
        """

        num_vars = [
        'OP_1', 'OP_2', 'OP_3'
        ] + [f'var_{i}' for i in range(1, 22)]

        # 24 features: 3 operations + 21 sensors

        X = []
        y = []

        grouped_engine_data = engine_data.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())

        for unit, group in grouped_engine_data:

            # print(f"--- Group: {unit} ---")
            # print(group) # print all rows in that group
            # print("\n")

            features = group[num_vars].values # all features in each engine, each feature is the op + sensor data

            # print("optional setting + sensor measurements:",features)          
            rul = group["RUL"].values                       

            num_cycles = len(features) # num_cycles for each engine

            # generate input sequences for each engine
            input_seqs =[]
            rul_value=[]
            for start in range(num_cycles - window_size + 1):
                end = start + window_size
                each_window = features[start:end]    

                target_rul = rul[end-1]

                # print("each_window:",each_window)   
                print("target_rul_for_cycle{} in engine{}:{}".format(start+window_size,unit,target_rul))    
                input_seqs.append(each_window)
                rul_value.append(target_rul)
            print("number of input seqs for engine {}:{}".format(unit,len(input_seqs)))

            X.extend(input_seqs)
            y.extend(rul)
        print("number of input seqs for all engines:{}".format(len(X)))

        return np.array(X), np.array(y)

    def train():
        pass

    def validation():
        pass

    def predict():
        pass


def plot_egine_cycles(df_train,index_names):

        max_time_cycles=df_train[index_names].groupby('unit_number').max()
        plt.figure(figsize=(20,50))
        ax=max_time_cycles['time_cycles'].plot(kind='barh',width=0.8, stacked=True,align='center')
        plt.title('Turbofan Engines LifeTime',fontweight='bold',size=30)
        plt.xlabel('Time cycle',fontweight='bold',size=20)
        plt.xticks(size=15)
        plt.ylabel('unit',fontweight='bold',size=20)
        plt.yticks(size=15)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    RNN = RNN_training()
    df_train,_,_=RNN.load_data()
    merged_data = RNN.Merged_with_RUL(df_train)
    # plot_egine_cycles(df_train,index_names)
    win_size =20
    X,y= RNN.create_input_sequnces(merged_data,win_size)
