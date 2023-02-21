import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import show_results as show

# create Dataset object and Dataloader
# The path to root directory, which contains UCF101 video files (not rawframes)
### Parameters ###############################################
subject_id = 1
subjects = {
    0 : "reconstruction",
    1 : "classification",
    2 : "interpolation"   
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
    #model = models.EncoderDecoder(3)
    
    encoder.eval()
    decoder.eval()
    #model.train()

    encoder.to(device)
    decoder.to(device)
    #model.to(device)

    # for using multi-gpu
    if lab_server_pc:
        print("Let's use multi-gpu!")
        encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
        decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])

    #show.plot_images(dataloader, encoder, decoder, device)

    # loss function
    loss_fn = nn.L1Loss() if subject_id==0 else nn.CrossEntropyLoss()

    log ={"loss":[]}
    loss_list = []

    # run test
    for i, batch in enumerate(tqdm(dataloader)):

        frame_batch = batch[0].to(device)

        # encode, and decode
        features = encoder(frame_batch)
        output = decoder(features)

        # calculate loss
        if subject_id == 0:
            loss = loss_fn(output, frame_batch)
        elif subject_id == 1:
            label_batch = batch[1].to(device)
            loss = loss_fn(output, label_batch).cpu()

        # clear cash
        del frame_batch
        if subject_id==1:
            del label_batch
        del features
        del output
        torch.cuda.empty_cache()
        
        if i % 100 == 0:
            log["loss"].append(float(loss))
        loss_list.append(loss)
        
    mean_value = np.mean(loss_list)
    std = np.std(loss_list)
    print("mean_value:\t", mean_value)
    print("std:\t", std)


    #x = range(len(log["loss"]))
    #plt.plot(x, log["loss"])
    #plt.show()