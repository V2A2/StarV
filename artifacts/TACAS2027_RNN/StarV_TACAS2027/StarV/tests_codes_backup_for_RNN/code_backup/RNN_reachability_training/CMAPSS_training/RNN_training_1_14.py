"""
Recurrent Layer Class
Qing Liu, 12/05/2025
"""
import random
from scipy.io import loadmat
import os
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



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
        sensor_names = [f'var_{i+1}' for i in range(0,21)]
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

        ''' Add a column 'RUL' for each row: RUL = (max cycle of that engine) - (current cycle)'''

        data = df_data.copy()

        max_cycles_per_engine = (data.groupby("unit_number")["time_cycles"].max()).rename("max_cycles")

        # cycles_per_engine = max_cycles_per_engine.reset_index().rename(columns={"time_cycles": "total_cycles"})
        # print("Number of cycles for each engine:")
        # print(cycles_per_engine)

        df_merged_data = data.join(max_cycles_per_engine, on="unit_number", how="left")

        df_merged_data["RUL"] = df_merged_data["max_cycles"] - df_merged_data["time_cycles"]
        df_merged_data['RUL'] = df_merged_data['RUL'].clip(upper=125)
        df_merged_data.drop(columns=["max_cycles"], inplace=True)
        

        print("\n Merged data with RUL:",df_merged_data.head(100))

        return df_merged_data


    def create_train_input_sequnces(self,train_data, window_size):
       
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

        train_data_1 = train_data.copy()

        all_features = list(train_data_1.columns[2:-1]) # 24 features: 3 operations + 21 sensors 


        # skip the most correlated features
        train_data_corr = train_data_1[all_features].corr().abs()
        upper_tri = train_data_corr.where(np.triu(np.ones(train_data_corr.shape),k=1).astype(bool))
        corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
        print("corr_features:", corr_features)
        train_data_1.drop(corr_features, axis=1, inplace=True)
        reamining_vars = [var for var in all_features if var not in corr_features]
        print("Remaining features after dropping correlated ones:", reamining_vars)


        # skip some conctant features, not useful for RUL prediction, not correalted with RUL
        const_vars = []
        for feature in reamining_vars:
            if train_data_1[feature].min() == train_data_1[feature].max():
                const_vars.append(feature)
                train_data_1.drop(feature,axis=1, inplace=True)

        selected_vars = [var for var in reamining_vars if var not in const_vars] 
        print("Dropped features with constant values:", const_vars)
        print("Remaining features:", selected_vars)
        print("Number of remaining features after dropping correlated ones and const ones:", len(selected_vars))
        print("Dropped train data shape:",train_data_1.shape)
        train_data_1.info()


        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train_data_1[selected_vars])
        train_data_1[selected_vars] = scaled_data
        print("Scaled train data samples:")
        print(train_data_1.head())

        final_selected_vars = ['time_cycles'] + selected_vars 
        print("Final selected vars:", final_selected_vars)

        X_train = []
        y_train = []
        X_val = []
        y_val =[]
        

        # group by engine unit
        grouped_engine_data = train_data_1.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())
        print("type of grouped_engine_data:",type(grouped_engine_data))


        # split data into training and validation sets
        engine_ids = train_data_1["unit_number"].unique()
        print("engine_ids:",engine_ids)
        val_engines = set(random.sample(list(engine_ids), 20))
        train_engines =  set(engine_ids) - val_engines

        print("Validation engines:", sorted(val_engines))
        print("Training engines:", sorted(train_engines))

        assert len(val_engines) == 20
        assert len(train_engines) == 80
        assert len(val_engines.intersection(train_engines)) == 0

        engine_ids_per_window = []
        num_windows_per_engine = [] 
        engine_order = [] 

        for unit, group in grouped_engine_data:

            engine_cycles = group[final_selected_vars].values # selected features in each engine
            # print('==============================',group.columns.tolist())

            # print(f"\n engine {unit} data samples:{engine_cycles[:10]}")

            # print("optional setting + sensor measurements:",features)   
            rul = group["RUL"].values                       

            num_cycles = len(engine_cycles) # num_cycles for each engine

            # generate input sequences for each engine
            input_seqs =[]
            rul_value=[]

            for start in range(num_cycles - window_size + 1):
                end = start + window_size
                each_window = engine_cycles[start:end]    
                target_rul = rul[end-1]
                # print("each_window:",each_window)   
                # print(f"target_rul_for_cycle{start+window_size} in engine{unit}:{target_rul}")    
                input_seqs.append(each_window) 
                rul_value.append(target_rul)
                engine_ids_per_window.append(unit)

            num_windows_per_engine.append(len(input_seqs))
            engine_order.append(unit)

            print(f"number of input seqs for train engine {unit}:{len(input_seqs)}")
 
            if unit in val_engines: # validation set
                X_val.extend(input_seqs)
                y_val.extend(rul_value)
            else:   # training set
                X_train.extend(input_seqs)
                y_train.extend(rul_value)


        print(f"number of input seqs for training: {len(X_train)}")
        # print(f"number of input seqs for validation: {len(X_val)}")

        return np.array(X_train), np.array(y_train),np.array(X_val),np.array(y_val),len(final_selected_vars)
    
    def create_test_input_sequnces(self,test_data, window_size):
       
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

        test_data_1 = test_data.copy()


        selected_vars = list(test_data_1.columns[2:-1]) # 24 features: 3 operations + 21 sensors 

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(test_data_1[selected_vars])
        test_data_1[selected_vars] = scaled_data

        print("Scaled test data samples:")
        print(test_data_1.head())


        final_selected_vars = ['time_cycles'] + selected_vars
        print("Final selected vars:", final_selected_vars)

        X_test = []
        # X_val = []
        # y_val =[]
        

        # group by engine unit
        grouped_engine_data = test_data_1.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())
        print("type of grouped_engine_data:",type(grouped_engine_data))


        # # split data into training and validation sets
        # engine_ids = engine_data_1["unit_number"].unique()
        # print("engine_ids:",engine_ids)
        # val_engines = set(random.sample(list(engine_ids), 20))
        # train_engines =  set(engine_ids) - val_engines

        # print("Validation engines:", sorted(val_engines))
        # print("Training engines:", sorted(train_engines))

        # assert len(val_engines) == 20
        # assert len(train_engines) == 80
        # assert len(val_engines.intersection(train_engines)) == 0

        engine_ids_per_window = []
        num_windows_per_engine = [] 
        engine_order = [] 

        for unit, group in grouped_engine_data:

            engine_cycles = group[final_selected_vars].values # selected features in each engine                   

            num_cycles = len(engine_cycles) # num_cycles for each engine

            # generate input sequences for each engine
            input_seqs =[]

            for start in range(num_cycles - window_size + 1):
                end = start + window_size
                each_window = engine_cycles[start:end]    
                # print("each_window:",each_window)   
                # print(f"target_rul_for_cycle{start+window_size} in engine{unit}:{target_rul}")    
                input_seqs.append(each_window) 
                engine_ids_per_window.append(unit)

            num_windows_per_engine.append(len(input_seqs))

            print(f"number of input seqs for test engine {unit}:{len(input_seqs)}")
 
            # if unit in val_engines: # validation set
            #     X_val.extend(input_seqs)
            #     y_val.extend(rul_value)
            # else:   # training set
            X_test.extend(input_seqs)
            all_windows = len(X_test)


        print(f"number of input seqs for training: {len(X_test)}")
        # print(f"number of input seqs for validation: {len(X_val)}")

        return np.array(X_test),len(final_selected_vars),np.array(engine_ids_per_window),np.array(num_windows_per_engine),all_windows


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
        # print(f"rnn_output:{rnn_out.shape}, hidden_state:{h}")
        last_hidden_state = h[-1]       
        # print(f"last_hidden_state_shape:{last_hidden_state.shape}")
        h_dropped = self.last_hidden_layer_dropout(last_hidden_state)
        # print(f"hidden_state_after_dropout_shape:{h_dropped.shape}")
        output = self.fc(h_dropped)
        # print(f"output_shape:{output.shape}")
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
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # self.t_total = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=4)
        # self.loss_fn = nn.SmoothL1Loss(beta=10.0)
        self.loss_fn = nn.MSELoss()
        
    
    def train(self):
        print("======================== Begin Training ========================")
        self.model.train()
        losses = []
        all_preds=[]
        all_targets =[]
        for idx, (x_batch,y_batch) in enumerate(self.train_loader):
            # print(type(x_batch))
            # print(len(x_batch))
            print("======================",x_batch[0],type(x_batch[0]), x_batch[0].shape)
            print("=============++++++++===========",y_batch[0],type(y_batch[0]), y_batch[0].shape)

            # x_batch = x_batch.to(self.device)
            # y_batch = y_batch.to(self.device)
            pred_rul = self.model(x_batch)
            print("pred_rul type:",type(pred_rul))
            loss = self.loss_fn(pred_rul,y_batch)
            print(f"pred_rul: {pred_rul}")
            print(f"true_rul: {y_batch}")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            self.optimizer.step()

            all_preds.append(pred_rul.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
            

            losses.append(loss.item())
            print("loss:",loss.item())
                                
        avg_loss = np.mean(losses)
        self.scheduler.step(avg_loss)
        print("Learning rate:", self.optimizer.param_groups[0]['lr'])

        return avg_loss
    
    def validate(self):
        print("======================== Begin Validation ========================")
        self.model.eval()
        losses = []
        all_preds=[]
        all_targets =[]
        with torch.no_grad():
            for idx, (x_batch,y_batch) in enumerate(self.val_loader):
                # x_batch = x_batch.to(self.device)
                # y_batch = y_batch.to(self.device)
                pred_rul = self.model(x_batch)
                loss = self.loss_fn(pred_rul,y_batch)
                losses.append(loss.item())
                all_preds.append(pred_rul.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
                                
        avg_loss = np.mean(losses)
        return avg_loss

    def save_model(self):
        for e in range(self.epochs):
            print(f"======== Epoch {e+1}/{self.epochs} ========")
            avg_train_losses = self.train()
            avg_val_losses = self.validate()
            print(f"=========================== - Train Loss {avg_train_losses:.3f} =========================")
            print(f"=========================== - Val Loss {avg_val_losses:.3f} =========================")

            # Save model checkpoint
            # torch.save(self.model.state_dict(), f"model_epoch_{e}.pth")


def predit(model, test_seqs):
    print("======================== Begin Validation ========================")
    model.eval()
    all_preds=[]
    with torch.no_grad():
        for w in test_seqs:
            pred_rul = model(w)
            # loss = self.loss_fn(pred_rul,y_batch)
            # losses.append(loss.item())
            all_preds.append(pred_rul.detach().cpu()).reshape(-1)
            # all_targets.append(y_batch.detach().cpu())
            # print(f"pred_rul: {pred_rul}")
            # print(f"true_rul: {y_batch}")
    return  np.concatenate(all_preds)


def pred_for_each_engine(self, preds, num_windows):
    splits = np.cumsum(num_windows)[:-1]
    per_engine = np.split(preds, splits)
    mean_pred_per_engine = []
    for p in per_engine:
        print("per_engine_pred_shape:",p.shape)
        mean_pred = p.mean()
        mean_pred_per_engine.append(mean_pred)
        mean_pred = np.array(mean_pred_per_engine)
        print("mean_pred_per_engine:",mean_pred)

    return mean_pred

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

def plot_corelation_heatmap(df_train):
    plt.figure(figsize=(12,10))
    threshold = 0.85
    corr = df_train.corr()
    mask = corr.where((abs(corr) >= threshold)).isna()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask,linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.show()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    set_seed(25) 
    data = RNN_dataset()
    df_train,df_test,y_test = data.load_data()
    merged_train_data = data.Merged_with_RUL(df_train)
    merged_train_data.info()

    # plot_egine_cycles(df_train,index_names = ['unit_number', 'time_cycles'])
    # plot_corelation_heatmap(df_train)

    win_size = 20
    X_train,y_train,X_val,y_val,num_features= data.create_train_input_sequnces(merged_train_data,win_size)
    X_test,_,_,_,all_windows = data.create_test_input_sequnces(df_test,win_size)
    print("input_train_seqs_shape:",X_train.shape)
    print("input_seqs_feature_type:",type(X_train))
    print("target_rul_shape:",y_train.shape)

    print("input_test_seqs_shape:",X_test.shape)
    print("input_seqs_feature_type:",type(X_test))

    # X = torch.tensor(input_seqs, dtype=torch.float32)
    # print("X_tshape:",X.shape)
    train_dataset = CMAPSS_Dataset(X_train,y_train)
    val_dataset = CMAPSS_Dataset(X_val,y_val)
    print("train_dataset_length:",train_dataset.__len__())
    # print("X_item:",X.__getitem__(0))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("train_datatloader:",len(train_loader))
    iter_train_loader = next(iter(train_loader))
    # print("iter_train_loader:",iter_train_loader[0])
    print("test_datatloader:",len(val_loader))


    model= RNN_model(input_size=num_features, hidden_size=32, fc_sizes=[64, 32, 16], dropout_prob=0.2)
    # model.init_weights()
    # output = model.forward(train_loader)
    # print("output:",output.detach().numpy())

    trainer = RNN_trainer(model, train_loader, val_loader, lr=1e-3, weight_decay=1e-4, epochs=50)
    trainer.save_model()

    preds = predit(model, X_test)
    print("preds_shape:",preds.shape)     
    mean_preds = pred_for_each_engine(trainer, preds, all_windows)
    print("mean_preds_per_engine:",mean_preds)
    true_rul = y_test["RUL"].values.reshape(-1)  # 100
    rmse = np.sqrt(mean_squared_error(true_rul, mean_preds))
    print("Test RMSE:", rmse)