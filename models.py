import torch


class Encoder(torch.nn.Module):
    def __init__(self, num_channel, kernel_size=5, pool_size=3):
        super(Encoder, self).__init__()
        
        # CNN for a single frame (spatial information)
        self.cnn1 = torch.nn.Conv3d(num_channel, 16, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn2 = torch.nn.Conv3d(16, 32, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn3 = torch.nn.Conv3d(32, 64, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn4 = torch.nn.Conv3d(64, 128, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn5 = torch.nn.Conv3d(128, 256, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn6 = torch.nn.Conv3d(256, 512, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.spatial_sequence = torch.nn.Sequential(
            self.cnn1,
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d(kernel_size=pool_size, stride=1),
            
            self.cnn2,
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d(kernel_size=pool_size, stride=1),
            
            self.cnn3,
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d(kernel_size=pool_size, stride=1),
            
            self.cnn4,
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d(kernel_size=pool_size, stride=1),
            
            self.cnn5,
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(),
            #torch.nn.MaxPool3d(kernel_size=pool_size, stride=1),
            #self.cnn6,
            #torch.nn.BatchNorm3d(512),
            #torch.nn.ReLU(),
        )
        
        # RNN for some frames converted to feature_vector (temporal information)
        self.temporal_cnn = torch.nn.RNN(input_size=512, hidden_size=512, num_layers=2)
        
    def forward(self, input_frames):
        feature = self.spatial_sequence(input_frames)
        #result = self.temporal_cnn(feature)
        return feature
        
        
class DecoderToFrames(torch.nn.Module):
    def __init__(self, num_channel, kernel_size=5, pool_size=3):
        super().__init__()
        
        # decode from hidden_state to 3d image
        self.decnn6 = torch.nn.ConvTranspose3d(512, 256, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn5 = torch.nn.ConvTranspose3d(256, 128, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn4 = torch.nn.ConvTranspose3d(128, 64, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn3 = torch.nn.ConvTranspose3d(64, 32, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn2 = torch.nn.ConvTranspose3d(32, 16, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn1 = torch.nn.ConvTranspose3d(16, num_channel, 3, stride=1, padding=1, padding_mode='zeros')

        self.decode_sequence = torch.nn.Sequential(
            #self.decnn6, torch.nn.BatchNorm3d(256), torch.nn.ReLU(),
            self.decnn5, torch.nn.BatchNorm3d(128), torch.nn.ReLU(), #torch.nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn4, torch.nn.BatchNorm3d(64), torch.nn.ReLU(), #torch.nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn3, torch.nn.BatchNorm3d(32), torch.nn.ReLU(), #torch.nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn2, torch.nn.BatchNorm3d(16), torch.nn.ReLU(), #torch.nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn1, torch.nn.BatchNorm3d(num_channel), torch.nn.ReLU(), #torch.nn.MaxUnpool3d(kernel_size=pool_size, stride=1)
        )
        
    def forward(self, input):
        result = self.decode_sequence(input)
        return result


class EncoderDecoder(torch.nn.Module):
    def __init__(self, num_channel):
        super().__init__()

        self.encoder = Encoder(num_channel)
        self.decoder = DecoderToFrames(num_channel)

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(features)
        return output
    

class DecoderToClassification(torch.nn.Module):
    def __init__(self, input_shape, class_indxs):
        super().__init__()
        
        
        # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames, C*H*W)
        self.flatten = torch.nn.Flatten(2,4)
        
        # (batch_size, num_frames, C*H*W) -> (batch_size, num_frames, len(class_indxs))
        # input_shape=(batch_size, C, num_frames, H, W)
        self.fc = torch.nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4]), out_features=len(class_indxs))
        
        # outputs probability vector
        #self.softmax = torch.nn.Softmax(dim=1)
        
    
    def forward(self, input):
        # input == features == encoder's output (shape = (batch_size, C_out, num_frames, H_out, W_out))
        
        input = input.permute(0,2,1,3,4)         # (batch_size, C, num_frames, H, W) -> (batch_size, num_frames, C, H, W)
        output = self.flatten(input)    # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames, C*H*W)
        output = self.fc(output)        # (batch_size, num_frames, C*H*W) -> (batch_size, num_frames, len(class_indxs))
        output = torch.sum(output, 1)   # (batch_size, num_frames, len(class_indxs)) -> (batch_size, len(class_indxs))
        #output = self.softmax(output)   # probability vectors
        
        return output
        
        
class DecoderMixReconClass(torch.nn.Module):
    # This class contains DecoderToFrames, DecoderToClassification instances which were trained.
    def __init__(self, path_recon, path_class):
        super().__init__()
        self.decoder_recon = torch.load(path_recon)
        self.decoder_class = torch.load(path_class)
        
        # Dataparallel
        if getattr(self.decoder_recon, 'device_ids', False):
            self.decoder_recon = self.decoder_recon.module
        if getattr(self.decoder_class, 'device_ids', False):
            self.decoder_class = self.decoder_class.module
        
        for param in self.decoder_recon.parameters():
            param.requires_grad = False
        for param in self.decoder_class.parameters():
            param.requires_grad = False
        
        
    def train(self):
        # In this case, only encoder will be trained. So decoders are set as "eval" mode.
        self.decoder_recon.eval()
        self.decoder_class.eval()
    
    def eval(self):
        self.decoder_recon.eval()
        self.decoder_class.eval()
    
    def to(self, device):
        self.decoder_recon.to(device)
        self.decoder_class.to(device)
        
    def forward(self, input):
        output_recon = self.decoder_recon(input)
        output_class = self.decoder_class(input)
        
        return [output_recon, output_class]
    
    
class DecoderAlternately(torch.nn.Module):
    def __init__(self, decoder1, decoder2):
        super().__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
    
    def train(self):
        # In this case, only encoder will be trained. So decoders are set as "eval" mode.
        self.decoder1.train()
        self.decoder2.train()
    
    def eval(self):
        self.decoder1.eval()
        self.decoder2.eval()
    
    def to(self, device):
        self.decoder1.to(device)
        self.decoder2.to(device)
        
    def forward(self, input, decoder_id):
        if decoder_id == 1:
            output = self.decoder1(input)
        elif decoder_id == 2:
            output = self.decoder2(input)
        elif decoder_id == 3:
            # train both decoders
            output = [self.decoder1(input), self.decoder2(input)]

        return output