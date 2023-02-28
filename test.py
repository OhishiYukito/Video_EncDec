import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import show_results as show

# create Dataset object and Dataloader
# The path to root directory, which contains UCF101 video files (not rawframes)
### Parameters ###############################################
subject_id = 1
subjects = {
    0 : "reconstruction",
    1 : "classification",
    2 : "interpolation",
    3 : "mix_recon-class",
}

input_H = 120
input_W = 160
batch_size = 4
lab_server_pc = False
##########################################################

if lab_server_pc:
    root_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucf101/UCF-101'
    ann_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucfTrainTestSplit'
else:
    root_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/videos'
    ann_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/ucfTrainTestSplit'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
print("device_count = {}".format(torch.cuda.device_count()))


tfs = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (C, T, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (input_H, input_W))),
])

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


dataset = UCF101(root= root_dir,
                 annotation_path= ann_dir,
                 frames_per_clip=5,
                 step_between_clips=3,
                 train=False,
                 transform=tfs,
                 num_workers=20
                 )


dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=custom_collate, 
                                         num_workers=1)
# data : ((batch_size, C, num_frames, H, W), (class_ids))
# data[1] has batch_size individual class_id.
# example : if batch_size=4, len(data[1])=4

with torch.no_grad():
    # create model instance
    folder_name = str(input_H)+'*'+str(input_W)
    folder_path = 'result/' + folder_name
    encoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_encoder_'+folder_name+'.pth')
    decoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_decoder_'+folder_name+'.pth')
    
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module
    
    encoder.eval()
    decoder.eval()

    encoder.to(device)
    decoder.to(device)

    # for using multi-gpu
    if lab_server_pc and subject_id!=1 and subject_id!=3:
        # When subject_id=1(classification), it's faster than the case of using multi-gpu to using single-gpu.
        # When subject_id=3(mix), the code will be more complex in the case of using multi-gpu. 
        print("Let's use multi-gpu!")
        encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
        decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])

    #show.plot_images(dataloader, encoder, decoder, device)

    # loss function
    if subject_id == 0:
        loss_fn = nn.L1Loss()
    elif subject_id == 1:
        loss_fn = nn.CrossEntropyLoss()
    elif subject_id == 3:
        loss_fn_recon = nn.L1Loss()
        loss_fn_class = nn.CrossEntropyLoss()

    log ={"loss":[]}
    loss_list = []

    # run test
    for i, batch in enumerate(tqdm(dataloader)):

        frame_batch = batch[0].to(device)
        label_batch = batch[1].to(device)

        # encode, and decode
        features = encoder(frame_batch)
        output = decoder(features)

        # calculate loss
        if subject_id == 0:
            loss = loss_fn(output, frame_batch)
        elif subject_id == 1:
            loss = loss_fn(output, label_batch)
        elif subject_id == 2:
            pass
        elif subject_id == 3:
            # loss_recon + loss_class
            loss_recon = loss_fn_recon(output[0], frame_batch)
            loss_class = loss_fn_class(output[1], label_batch)
            loss = [loss_recon, loss_class]
            
        # clear cash
        del frame_batch
        del label_batch
        del features
        del output
        torch.cuda.empty_cache()
        
        if i % 100 == 0:
            log["loss"].append(float(loss))
        loss_list.append(loss)
        
    with open(folder_path +'/'+ subjects[subject_id]+'_evaluate_'+folder_name+'.pkl', 'wb') as f:
        pkl.dump(loss_list, f)
        
    mean_value = np.mean(loss_list)
    std = np.std(loss_list)
    print("mean_value:\t", mean_value)
    print("std:\t", std)
    
    
    if subject_id==3:
        x = range(0, len(log["loss"])*100, 100)
        y_recon = [data[0] for data in loss_list]
        y_class = [data[1] for data in loss_list]
        y_sum = [data[0]+data[1] for data in loss_list]
        plt.plot(x, y_recon)
        plt.plot(x, y_class)
    #x = range(len(log["loss"]))
    #plt.plot(x, log["loss"])
    #plt.show()