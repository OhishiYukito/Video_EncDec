import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_channel, kernel_size=3, pool_size=3):
        super().__init__()
        
        # CNN for a single frame (spatial information)
        self.cnn1 = nn.Conv3d(num_channel, 16, 3, stride=1, padding=1, padding_mode='zeros')
        self.cnn2 = nn.Conv3d(16, 32, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn3 = nn.Conv3d(32, 64, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn4 = nn.Conv3d(64, 128, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn5 = nn.Conv3d(128, 256, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.cnn6 = nn.Conv3d(256, 512, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.spatial_sequence = nn.Sequential(
            self.cnn1, nn.BatchNorm3d(16), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn2, nn.BatchNorm3d(32), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn3, nn.BatchNorm3d(64), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn4, nn.BatchNorm3d(128), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn5, nn.BatchNorm3d(256), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            #self.cnn6, #nn.BatchNorm3d(512), #nn.ReLU(),
        )
        
        # RNN for some frames converted to feature_vector (temporal information)
        self.temporal_cnn = nn.RNN(input_size=512, hidden_size=512, num_layers=2)
        
    def forward(self, input_frames):
        feature = self.spatial_sequence(input_frames)
        #result = self.temporal_cnn(feature)
        return feature
        
        
class DecoderToReconstruction(nn.Module):
    def __init__(self, num_channel, kernel_size=3, pool_size=3):
        super().__init__()
        
        # decode from hidden_state to 3d image
        self.decnn6 = nn.ConvTranspose3d(512, 256, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn5 = nn.ConvTranspose3d(256, 128, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn4 = nn.ConvTranspose3d(128, 64, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn3 = nn.ConvTranspose3d(64, 32, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn2 = nn.ConvTranspose3d(32, 16, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.decnn1 = nn.ConvTranspose3d(16, num_channel, 3, stride=1, padding=1, padding_mode='zeros')

        self.decode_sequence = nn.Sequential(
            #self.decnn6, nn.BatchNorm3d(256), nn.ReLU(),
            self.decnn5, nn.BatchNorm3d(128), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn4, nn.BatchNorm3d(64), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn3, nn.BatchNorm3d(32), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn2, nn.BatchNorm3d(16), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn1, nn.BatchNorm3d(num_channel), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1)
        )
        
    def forward(self, input):
        result = self.decode_sequence(input)
        return result


class EncoderDecoder(nn.Module):
    def __init__(self, num_channel):
        super().__init__()

        self.encoder = Encoder(num_channel)
        self.decoder = DecoderToReconstruction(num_channel)

    def forward(self, input):
        features = self.encoder(input)
        output = self.decoder(features)
        return output
    

class DecoderToClassification(nn.Module):
    def __init__(self, input_shape, class_indxs):
        super().__init__()
        
        
        # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames, C*H*W)
        self.flatten = nn.Flatten(2,4)
        
        # (batch_size, num_frames, C*H*W) -> (batch_size, num_frames, len(class_indxs))
        # input_shape=(batch_size, C, num_frames, H, W)
        self.fc1 = nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4]), out_features=(input_shape[1]*input_shape[3]*input_shape[4])//2)
        self.fc2 = nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4])//2, out_features=(input_shape[1]*input_shape[3]*input_shape[4])//4)
        self.fc3 = nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4])//4, out_features=(input_shape[1]*input_shape[3]*input_shape[4])//8)
        self.fc4 = nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4])//8, out_features=(input_shape[1]*input_shape[3]*input_shape[4])//16)
        self.fc5 = nn.Linear(in_features=(input_shape[1]*input_shape[3]*input_shape[4])//16, out_features=len(class_indxs))
        # outputs probability vector
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, input):
        # input == features == encoder's output (shape = (batch_size, C_out, num_frames, H_out, W_out))
        
        input = input.permute(0,2,1,3,4)         # (batch_size, C, num_frames, H, W) -> (batch_size, num_frames, C, H, W)
        output = self.flatten(input)    # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames, C*H*W)
        
        output = self.fc1(output)        # (batch_size, num_frames, C*H*W) -> (batch_size, num_frames, len(class_indxs))
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        
        output = torch.sum(output, 1)   # (batch_size, num_frames, len(class_indxs)) -> (batch_size, len(class_indxs))
        output = self.softmax(output)   # probability vectors
        
        return output
        
        
class DecoderMixReconClass(nn.Module):
    # This class contains DecoderToReconstruction, DecoderToClassification instances which were trained.
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
    
    
class DecoderAlternately(nn.Module):
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
    
    
    
# https://github.com/pytorch/vision/blob/9e474c3c46c0871838c021093c67a9c7eb1863ea/torchvision/models/video/resnet.py#L36   
class Conv2Plus1D(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=2,
                 padding=1):
        super().__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(padding, 0, 0),      # if stride=(2,1,1), the number of input frames will be decrease, and couldn't convolution
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride
    
class ConvTranspose2Plus1D(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=2,
                 padding=1):
        super().__init__(
            nn.ConvTranspose3d(in_channels, mid_channels, kernel_size=(3,1,1),
                               stride=(1,1,1), padding=(padding,0,0),
                               bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=(1,3,3),
                               stride=(1,stride,stride), padding=(0,padding,padding), output_padding=(0,1,1),
                               bias=False)
        )
        
    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride
    
    
class Encoder2Plus1D(nn.Module):
    def __init__(self, num_channel, kernel_size=3, pool_size=1):
        super().__init__()
        
        # CNN for a single frame (spatial information)
        self.cnn1 = Conv2Plus1D(num_channel, 16, (16-num_channel)//2+num_channel, stride=2, padding=1)
        self.cnn2 = Conv2Plus1D(16, 32, 24, stride=2, padding=1)
        self.cnn3 = Conv2Plus1D(32, 64, 48, stride=2, padding=1)
        self.cnn4 = Conv2Plus1D(64, 128, 96, stride=2, padding=1)
        self.cnn5 = Conv2Plus1D(128, 256, 192, stride=2, padding=1)
        self.cnn6 = Conv2Plus1D(256, 512, 384, stride=2, padding=1)
        self.spatial_sequence = nn.Sequential(
            self.cnn1, nn.BatchNorm3d(16), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn2, nn.BatchNorm3d(32), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn3, nn.BatchNorm3d(64), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn4, nn.BatchNorm3d(128), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            self.cnn5, nn.BatchNorm3d(256), nn.ReLU(), #nn.MaxPool3d(kernel_size=pool_size, stride=1),
            #self.cnn6, #nn.BatchNorm3d(512), #nn.ReLU(),
        )
        
        # RNN for some frames converted to feature_vector (temporal information)
        self.temporal_cnn = nn.RNN(input_size=512, hidden_size=512, num_layers=2)
        
    def forward(self, input_frames):
        feature = self.spatial_sequence(input_frames)
        #result = self.temporal_cnn(feature)
        return feature
    
class DecoderToReconstruction2Plus1D(nn.Module):
    def __init__(self, num_channel, kernel_size=5, pool_size=3):
        super().__init__()
        
        # decode from hidden_state to 3d image
        self.decnn6 = ConvTranspose2Plus1D(512, 256, 384, stride=2, padding=1)
        self.decnn5 = ConvTranspose2Plus1D(256, 128, 192, stride=2, padding=1)
        self.decnn4 = ConvTranspose2Plus1D(128, 64, 96, stride=2, padding=1)
        self.decnn3 = ConvTranspose2Plus1D(64, 32, 48, stride=2, padding=1)
        self.decnn2 = ConvTranspose2Plus1D(32, 16, 24, stride=2, padding=1)
        self.decnn1 = ConvTranspose2Plus1D(16, num_channel, (16-num_channel)//2+num_channel, stride=2, padding=1)

        self.decode_sequence = nn.Sequential(
            #self.decnn6, nn.BatchNorm3d(256), nn.ReLU(),
            self.decnn5, nn.BatchNorm3d(128), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn4, nn.BatchNorm3d(64), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn3, nn.BatchNorm3d(32), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn2, nn.BatchNorm3d(16), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1),
            self.decnn1, nn.BatchNorm3d(num_channel), nn.ReLU(), #nn.MaxUnpool3d(kernel_size=pool_size, stride=1)
        )
        
    def forward(self, input):
        result = self.decode_sequence(input)
        return result