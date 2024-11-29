from torch import nn

class SemanticDecoder_MLP(nn.Module):
    def __init__(self, feature_out_dim):
        super().__init__()
        self.output_dim = feature_out_dim
        #self.fc1 = nn.Linear(16, 128).cuda()
        #self.fc1 = nn.Linear(128, 128).cuda()
        #self.fc2 = nn.Linear(128, self.output_dim).cuda()

        #self.fc0 = nn.Linear(16, 16).cuda()
        #self.fc1 = nn.Linear(16, 32).cuda()
        #self.fc2 = nn.Linear(32, 64).cuda()
        #self.fc3 = nn.Linear(128, 128).cuda()
        self.fc4 = nn.Linear(128, 256).cuda()

    def forward(self, x):
        input_dim, h, w = x.shape
        x = x.permute(1,2,0).contiguous().view(-1, input_dim) #(16,48,64)->(48,64,16)->(48*64,16)
        #x = torch.relu(self.fc0(x))
        #x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = torch.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous()
        return x

class SemanticDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):
        
        x = self.conv(x)

        return x