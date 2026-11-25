import torch.nn as nn 
import torch 

class Critic(nn.Module) : 
    def __init__(self,hidden_layers_num=2, hidden_layer_size=128,input_size=25, output_layer_size=1) : 
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers_num = hidden_layers_num 
        self.output_layer_size = output_layer_size
        hidden_layers = [nn.Linear(hidden_layer_size,hidden_layer_size),nn.ReLU() for _ in range(hidden_layers_num-1)]
        self.network = nn.Sequential(
            nn.Linear(input_size,hidden_layer_size), 
            nn.ReLU(), 
            *hidden_layers , 
            nn.Linear(hidden_layer_size), 
            nn.ReLU()
        ) 
    def forward(self,x) :
        return self.network(x)

class Policy(nn.Module) :
    def __init__(self,encoder_input_size=25,encoder_hidden_layers_num=2, encoder_hidden_layer_size=128, encoder_output=64, 
                 transformer_layers_num=2,d_model=64,n_head=64,dropout=0.0) :
        hidden_layers = [nn.Linear(encoder_hidden_layer_size,encoder_hidden_layer_size),nn.ReLU() for _ in range(encoder_hidden_layers_num -1)]
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size,encoder_hidden_layer_size), 
            *hidden_layers, 
            nn.Linear(encoder_hidden_layer_size,encoder_output), 
            nn.ReLU()
        ) 
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model,droput=dropout,batch_first=True, norm_first=True) 
        self.transformer_decoder = nn.TransformerEncoder(layer, num_layers=transformer_layers_num)
        self.transformer_head = nn.Linear(d_model,1)
    def forward(self,x) : 
        h = self.encoder(x) 
        z = self.transformer_decoder(h) 
        return self.head(z).squeeze(-1)

# def train_with_PPO(x) : 

