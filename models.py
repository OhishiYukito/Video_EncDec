import torch

class Encoder(torch.nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        
        # CNN for a single frame (spatial information)
        self.cnn1 = torch.nn.Conv3d(num_channel, 16, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn2 = torch.nn.Conv3d(16, 32, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn3 = torch.nn.Conv3d(32, 64, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn4 = torch.nn.Conv3d(64, 128, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn5 = torch.nn.Conv3d(128, 256, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn6 = torch.nn.Conv3d(256, 512, 3, stride=1, padding=1, padding_mode='zeros')
        self.spatial_sequence = torch.nn.Sequential(
            self.cnn1,
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d((1,3,3), stride=1),
            self.cnn2,
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d((1,3,3), stride=1),
            self.cnn3,
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d((1,3,3), stride=1),
            self.cnn4,
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d((1,3,3), stride=1),
            self.cnn5,
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d((1,3,3), stride=1),
            self.cnn6,
            torch.nn.ReLU(),
        )
        
        # RNN for some frames converted to feature_vector (temporal information)
        self.temporal_cnn = torch.nn.RNN(input_size=512, hidden_size=512, num_layers=2)
        
    def forward(self, input_frames):
        feature = self.spatial_sequence(input_frames)
        #result = self.temporal_cnn(feature)
        return feature
        
        
class DecoderToFrames(torch.nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        
        # decode from hidden_state to 3d image
        self.decnn6 = torch.nn.ConvTranspose3d(512, 256, 3, stride=1, padding=1, padding_mode='zeros')
        self.decnn5 = torch.nn.ConvTranspose3d(256, 128, 3, stride=1, padding=1, padding_mode='zeros')
        self.decnn4 = torch.nn.ConvTranspose3d(128, 64, 3, stride=1, padding=1, padding_mode='zeros')
        self.decnn3 = torch.nn.ConvTranspose3d(64, 32, 3, stride=1, padding=1, padding_mode='zeros')
        self.decnn2 = torch.nn.ConvTranspose3d(32, 16, 3, stride=1, padding=1, padding_mode='zeros')
        self.decnn1 = torch.nn.ConvTranspose3d(16, num_channel, 3, stride=1, padding=1, padding_mode='zeros')

        self.decode_sequence = torch.nn.Sequential(
            self.decnn6, torch.nn.ReLU(),
            self.decnn5, torch.nn.ReLU(),
            self.decnn4, torch.nn.ReLU(),
            self.decnn3, torch.nn.ReLU(),
            self.decnn2, torch.nn.ReLU(),
            self.decnn1
        )
        
    def forward(self, input):
        result = self.decode_sequence(input)
        return result