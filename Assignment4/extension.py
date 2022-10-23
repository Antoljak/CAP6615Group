class CNet(nn.Module):
    def __init__(self, linear_input=60):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=20,padding=2,padding_mode='replicate', kernel_size=5, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(linear_input, 4)
        
    def forward(self, x):
        if(test==True):
            x = x.permute(0,2,1)
        else:
            x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        
        x =  self.ReLU( self.max_pool1(x) ) 
        
        x = x.view(x.size(0), -1)
        x = self.ReLU(self.linear1(x))
        x = x.unsqueeze(1)
        return x


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.cnn = CNet()
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.ReLU = nn.ReLU()
        
    
    def forward(self, x):
       
        x = self.cnn(x) 
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
    
        out = self.fc(out)
    
        return out[:, -1, :], hidden
    
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    #correlation algorithm
    def corrcoef(dataset):
        sampleNum = 100
        for i in range(sampleNum-1 ,len(dataset)):
    # Pearson  r
            x = dataset[(i-sampleNum+1):(i+1), 0]
            y = dataset[(i-sampleNum+1):(i+1), 1]
            r = np.corrcoef(x, y)
            dataset[i][2] = r[0,1] 
        return dataset