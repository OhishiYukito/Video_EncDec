
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import tools.initial_process as init
import tools.show_functions as show
### Parameters ###############################################
subject_id = 1
subjects = {
    0 : "reconstruction",
    1 : "classification",
    2 : "interpolation",
    3 : "mix_recon-class",
    4 : "alternately_recon-class",
}

#input_H = 120
#input_W = 160
input_H = 192
input_W = 256
batch_size = 8
lab_server_pc = True


base_model_id = 3
folder_name_list = {
    0 : "Conv3d",       # if base_model is Conv3d
    1 : "(2+1)D_without_softmax",       # if base_model is Conv2Plus1D
    2 : "(2+1)D",
    3 : "(2+1)D_DecToClass_fc5",
}
load_model_epoch = 5
##########################################################


class_indxs, device, dataloader = init.initial_process(lab_server_pc,
                                                       subject_id, 
                                                       input_H, 
                                                       input_W, 
                                                       batch_size, 
                                                       train=False)

# load models
folder_name = folder_name_list[base_model_id]
folder_path = 'result/' + folder_name
input_shape = str(input_H)+'*'+str(input_W)
file_path_incomplete = folder_path + '/' + subjects[subject_id]
if load_model_epoch==1:
    file_path_last_part = input_shape
else:
    file_path_last_part = input_shape + '_'+str(load_model_epoch).zfill(2)+'epoch'
encoder = torch.load(file_path_incomplete +'_encoder_'+ file_path_last_part+'.pth')
decoder = torch.load(file_path_incomplete +'_decoder_'+ file_path_last_part+'.pth')

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
    #show.plot_reconstruction(dataloader, encoder, decoder, device)
    show.plot_animation(dataloader, 
                        encoder, 
                        decoder, 
                        device, 
                        base_model_name= folder_name_list[base_model_id],
                        subject_name= subjects[subject_id])
elif subject_id == 1:
    show.plot_classification(dataloader, encoder, decoder, device)
elif subject_id == 2:
    pass
elif subject_id == 3 or subject_id == 4:
    show.plot_recon_class(dataloader, encoder, decoder, device, subject_id)