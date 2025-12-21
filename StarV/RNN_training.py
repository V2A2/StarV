"""
Recurrent Layer Class
Qing Liu, 12/05/2025
"""
from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class RNN_dataset(object):

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

        print("all_train_data_shape:",df_train.shape)
        print("all_test_data_shape:",df_test.shape)
        print("all_test_URL_shape:",y_test.shape)

        # train_samples = df_train.head(10)
        # pd.set_option('display.max_column', 30)
        # print("train_data_samples:",train_samples)/

        return df_train,df_test,y_test


    def Merged_with_RUL(self, df_data):

        ''' Add additinal column for RUL for each engine unit cycle'''

        total_cycles_per_engine = (df_data.groupby("unit_number")["time_cycles"].max())
        cycles_per_engine = total_cycles_per_engine.reset_index().rename(columns={"time_cycles": "total_cycles"})
        print("Number of cycles for each engine:")
        print(cycles_per_engine)

        df_merged_data = df_data.merge(cycles_per_engine, on="unit_number", how="left")

        df_merged_data["RUL"] = df_merged_data["total_cycles"] - df_merged_data["time_cycles"]

        df_merged_data = df_merged_data.drop("total_cycles", axis=1) 
        
        # df_merged_data = df_merged_data.drop("total_cycles", axis=1) 

        print("\n Merged data with RUL:",df_merged_data)

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

        # group by engine unit
        grouped_engine_data = engine_data.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())

        for unit, group in grouped_engine_data:

            # print(f"--- Group: {unit} ---\n")
            # print(group) # print all rows in that group

            features = group[num_vars].values # all features in each engine, each feature is the optional setting(3)+ sensor measurements(21)

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
                # print("target_rul_for_cycle{} in engine{}:{}".format(start+window_size,unit,target_rul))    
                input_seqs.append(each_window)
                rul_value.append(target_rul)
            # print("number of input seqs for engine {}:{}".format(unit,len(input_seqs)))

            X.extend(input_seqs) # including RUL
            y.extend(rul_value)
        print(f"X:{X[0]}")
        print(f"y:{y[0]}")
        print("number of input seqs for all engines:{}".format(len(X)))

        return np.array(X), np.array(y)
    

class CMAPSS_Dataset(Dataset):
    def __init__(self, X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        inputs = self.X[idx].squeeze(0)
        rul = self.y[idx].squeeze(0)
        return inputs, rul


class RNN_model(nn.Module):
    def __init__(self, input_size: int, hidden_size:int, fc_sizes:list,dropout_prob:float):

        super().__init__()
    

        # Vanilla RNN with ReLu
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='relu',
            batch_first=True,
            dropout=0
        )

        # apply dropout to the last hidden layer
        self.last_hidden_layer_dropout = nn.Dropout(dropout_prob)

        # Fully connected layers with ReLu
        fc_layers = []
        prev_layer = hidden_size
        for s in fc_sizes:
            fc_layers.append(nn.Linear(prev_layer, s))
            fc_layers.append(nn.ReLU())   
            fc_layers.append(nn.Dropout(0.2))  # FC dropout layer
            prev_layer = s
        
        output_layer = nn.Linear(prev_layer, 1)  # output RUL

        fc_layers.append(output_layer)
        self.fc = nn.Sequential(*fc_layers)

        self.init_weights()
    
    def init_weights(self):
        # RNN initialization
        for name, param in self.rnn.named_parameters():
            print(f"RNN_Name: {name} ---- Shape: {param.shape}")
            if "weight_ih" in name:
                # nn.init.xavier_uniform_(param) # weight init from xavier uniform(-a,a)
                nn.init.kaiming_uniform_(param) 
                # print(f"weight_ih_init:{param}")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # FC layers initialization
        for i, layer in enumerate(self.fc):
            for name, param in layer.named_parameters():
                if isinstance(layer, nn.Linear):
                    # print(f"FC_layer {i}: {layer} ---- Shape: {param.shape}")
                    # if isinstance(param, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
                    # print(f"FC_layer {i} weight initialization:{layer.weight}")
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        print(f"input_shape in forward:{x.shape}")
        rnn_out, h = self.rnn(x) # output: tensor of shape (L,D∗Hout)(L,D∗Hout​)  h: tensor of shape (D*num_layers,Hout​), D = 2 if bidirectional=True otherwise 1
        print(f"rnn_output:{rnn_out.shape}, hidden_state:{h}")
        last_hidden_state = h[-1]       
        print(f"last_hidden_state_shape:{last_hidden_state.shape}")
        h_dropped = self.last_hidden_layer_dropout(last_hidden_state)
        print(f"hidden_state_after_dropout_shape:{h_dropped.shape}")
        output = self.fc(h_dropped)
        print(f"output_shape:{output.shape}")
        output = output.squeeze(-1)        
        return output


class RNN_trainer(object):
    def __init__(self, model:nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, 
                 lr: float,weight_decay: float, epochs: int):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.device = device
        self.epochs = epochs
        # self.model_save_path = model_save_path
        self.lr = lr
        self.weight_decay = weight_decay

         # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # self.t_total = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=5)
        self.loss_fn = nn.SmoothL1Loss(beta=10.0)
        
    
    def train(self):
        print("======================== Begin Training model ========================")
        avg_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            all_preds=[]
            all_targets =[]
            for idx, (x_batch,y_batch) in enumerate(self.train_loader):
                # print(type(x_batch))
                # print(len(x_batch))
                # print(type(x_batch[0]), x_batch[0].shape)

                # x_batch = x_batch.to(self.device)
                # y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred_rul = self.model(x_batch)
                loss = self.loss_fn(pred_rul,y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.optimizer.step()

                print("pred_rul:",pred_rul)
                all_preds.append(pred_rul.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
                print("true_rul:",y_batch)
                losses.append(loss.item())
                print("loss:",loss.item())
                

            avg_loss = float(np.mean(losses))
            avg_losses.append(avg_loss)
            print("Pred mean:", pred_rul.mean(), "True mean:", y_batch.mean())
        return avg_losses

        



    def validation(self):
        pass

    # def predict():
    #     pass


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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    set_seed(25) 
    data = RNN_dataset()
    df_train,_,_=data.load_data()
    merged_data = data.Merged_with_RUL(df_train)
    # plot_egine_cycles(df_train,index_names = ['unit_number', 'time_cycles'])
    win_size =20
    input_seqs,target_rul= data.create_input_sequnces(merged_data,win_size)
    print("input_seqs_shape:",input_seqs.shape)
    print("target_rul_shape:",target_rul.shape)

    # X = torch.tensor(input_seqs, dtype=torch.float32)
    # print("X_tshape:",X.shape)
    train_dataset = CMAPSS_Dataset(input_seqs,target_rul)
    print("train_dataset_length:",train_dataset.__len__())
    # print("X_item:",X.__getitem__(0))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # train_dataloader = train_dataset.get_data_loader(batch_size=32)
    print("train_datatloader:",len(train_loader))
    iter_train_loader = next(iter(train_loader))
    # print("iter_train_loader:",iter_train_loader[0])


    model= RNN_model(input_size=24, hidden_size=32, fc_sizes=[64, 32, 16], dropout_prob=0.3)
    # model.init_weights()
    # output = model.forward(train_loader)
    # print("output:",output.detach().numpy())

    trainer = RNN_trainer(model, train_loader, train_loader, lr=1e-3, weight_decay=1e-4, epochs=50)
    trainer.train()
