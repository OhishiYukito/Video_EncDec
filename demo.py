
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import tools.initial_process as init
import tools.show_functions as show
### Parameters ###############################################
subject_id = 4
subjects = {
    0 : "reconstruction",
    1 : "classification",
    2 : "interpolation",
    3 : "mix_recon-class",
    4 : "alternately_recon-class",
}

input_H = 120
input_W = 160
batch_size = 8
lab_server_pc = True
##########################################################


class_indxs, device, dataloader = init.initial_process(lab_server_pc,
                                                       subject_id, 
                                                       input_H, 
                                                       input_W, 
                                                       batch_size, 
                                                       train=False)

# load models
folder_name = str(input_H)+'*'+str(input_W)
folder_path = 'result/' + folder_name
encoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_encoder_'+folder_name+'.pth')
decoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_decoder_'+folder_name+'.pth')
print("encoder : ", subjects[subject_id]+'_encoder_'+folder_name+'.pth')
print("decoder : ", subjects[subject_id]+'_decoder_'+folder_name+'.pth')

if getattr(encoder, 'device_ids', False):
    encoder = encoder.module
    decoder = decoder.module

encoder.eval()
decoder.eval()

encoder.to(device)
decoder.to(device)

# for using multi-gpu
if lab_server_pc and subject_id!=1 and subject_id!=3 and subject_id!=4:
    print("Let's use multi-gpu!")
    encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
    decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])

#show.plot_images(dataloader, encoder, decoder, device)
if subject_id == 0:
    show.plot_reconstruction(dataloader, encoder, decoder, device)
elif subject_id == 1:
    show.plot_classification(dataloader, encoder, decoder, device)
elif subject_id == 2:
    pass
elif subject_id == 3 or subject_id == 4:
    show.plot_recon_class(dataloader, encoder, decoder, device, subject_id)