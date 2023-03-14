import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tools.initial_process as init

# create Dataset object and Dataloader
# The path to root directory, which contains UCF101 video files (not rawframes)
### Parameters ###############################################
subject_id = 1
subjects = {
    0 : "reconstruction",
    1 : "classification",
    2 : "interpolation",
    3 : "mix_recon-class",
    4 : "alternately_recon-class",
}

input_H = 192
input_W = 256
batch_size = 8
lab_server_pc = True

base_model_id = 1
folder_name_list = {
    0 : "Conv3d",       # if base_model is Conv3d
    1 : "(2+1)D",       # if base_model is Conv2Plus1D
}

##########################################################

class_indxs, device, dataloader = init.initial_process(lab_server_pc,
                                                       subject_id, 
                                                       input_H, 
                                                       input_W, 
                                                       batch_size, 
                                                       train=False)

with torch.no_grad():
    # create model instance
    folder_name = folder_name_list[base_model_id]
    input_shape = str(input_H)+'*'+str(input_W)
    folder_path = 'result/' + folder_name
    encoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_encoder_'+input_shape+'.pth')
    decoder = torch.load(folder_path +'/'+ subjects[subject_id]+'_decoder_'+input_shape+'.pth')
    
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module
    
    encoder.eval()
    decoder.eval()

    encoder.to(device)
    decoder.to(device)

    # for using multi-gpu
    if lab_server_pc and subject_id!=1 and subject_id!=3 and subject_id!=4:
        # When subject_id=1(classification), it's faster than the case of using multi-gpu to using single-gpu.
        # When subject_id=3(mix), the code will be more complex in the case of using multi-gpu. 
        print("Let's use multi-gpu!")
        encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
        decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])

    # loss function
    if subject_id == 0:
        loss_fn = nn.L1Loss()
    elif subject_id == 1:
        loss_fn = nn.CrossEntropyLoss()
    elif subject_id == 3 or subject_id == 4:
        loss_fn_recon = nn.L1Loss()
        loss_fn_class = nn.CrossEntropyLoss()

    #log ={"loss":[]}
    loss_list = []

    # run test
    for i, batch in enumerate(tqdm(dataloader)):

        frame_batch = batch[0].to(device)
        label_batch = batch[1].to(device)

        # encode, and decode
        features = encoder(frame_batch)
        if subject_id == 4:
            # get 2 decoder's output
            output = decoder(features, 3)
        else:
            output = decoder(features)

        # calculate loss
        if subject_id == 0:
            loss = loss_fn(output, frame_batch)
            loss = loss.cpu()
        elif subject_id == 1:
            loss = loss_fn(output, label_batch)
            loss = loss.cpu()
        elif subject_id == 2:
            pass
        elif subject_id == 3 or subject_id==4:
            # loss_recon + loss_class
            loss_recon = loss_fn_recon(output[0], frame_batch)
            loss_class = loss_fn_class(output[1], label_batch)

            loss = [loss_recon.cpu(), loss_class.cpu()]

        #if i % 100 == 0:
            #log["loss"].append(float(loss))
        loss_list.append(loss)    

        # clear cash
        del frame_batch
        del label_batch
        del features
        del output
        torch.cuda.empty_cache()
        

        
    with open(folder_path +'/'+ subjects[subject_id]+'_evaluate_'+input_shape+'.pkl', 'wb') as f:
        pkl.dump(loss_list, f)
        
    if subject_id==3 or subject_id==4:
        loss_recon = [loss[0] for loss in loss_list]
        mean_value = np.mean(loss_recon)
        std = np.std(loss_recon)
        print("mean_value(recon):\t", mean_value)
        print("std(recon):\t", std)

        loss_class = [loss[1] for loss in loss_list]
        mean_value = np.mean(loss_class)
        std = np.std(loss_class)
        print("mean_value(class):\t", mean_value)
        print("std(class):\t", std)

        loss_list = [loss[0]+loss[1] for loss in loss_list]

    mean_value = np.mean(loss_list)
    std = np.std(loss_list)
    print("mean_value:\t", mean_value)
    print("std:\t", std)
    
    #if subject_id==3:
    #    x = range(0, len(log["loss"])*100, 100)
    #    y_recon = [data[0] for data in loss_list]
    #    y_class = [data[1] for data in loss_list]
    #    y_sum = [data[0]+data[1] for data in loss_list]
    #    plt.plot(x, y_recon)
    #    plt.plot(x, y_class)
    
    #x = range(len(log["loss"]))
    #plt.plot(x, log["loss"])
    #plt.show()