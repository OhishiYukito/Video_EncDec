import models
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
import tools.initial_process as init

### Parameters ###############################################
subject_id = 4
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
                                                       train=True)

# create model instance
encoder = models.Encoder(3) if base_model_id==0 else models.Encoder2Plus1D(3)
if subject_id == 0:
    # Reconstruction
    decoder = models.DecoderToReconstruction(3) if base_model_id==0 else models.DecoderToReconstruction2Plus1D(3)
elif subject_id == 1:
    # Classification
    # get feature's shape
    encoder.eval()
    example = next(iter(dataloader))[0]
    example = encoder(example)
    decoder = models.DecoderToClassification(input_shape= example.shape, class_indxs= class_indxs)
elif subject_id == 2:
    # Interpolation
    pass
elif subject_id == 3:
    # Mix (Reconstruction & Classification)
    folder_name = folder_name_list[base_model_id]
    input_shape = str(input_H)+'*'+str(input_W)
    folder_path = 'result/' + folder_name
    path_recon = folder_path +'/'+ subjects[0]+'_decoder_'+input_shape+'.pth'
    path_class = folder_path +'/'+ subjects[1]+'_decoder_'+input_shape+'.pth'
    decoder = models.DecoderMixReconClass(path_recon, path_class)
elif subject_id == 4:
    # train Reconstruction & Classification Decoders and Encoder Alternately 
    decoder1 = models.DecoderToReconstruction(3) if base_model_id==0 else models.DecoderToReconstruction2Plus1D(3)
    encoder.eval()
    example = next(iter(dataloader))[0]
    example = encoder(example)
    decoder2 = models.DecoderToClassification(input_shape= example.shape, class_indxs= class_indxs)
    decoder = models.DecoderAlternately(decoder1, decoder2)

    
#model = models.EncoderDecoder(3)
 
encoder.train()
decoder.train()
#model.train()

encoder.to(device)
decoder.to(device)
#model.to(device)

# for using multi-gpu
if lab_server_pc and subject_id!=1 and subject_id!=3 and subject_id!=4:
    # When subject_id=1(classification), it's faster than the case of using multi-gpu to using single-gpu.
    # When subject_id=3(mix), the code will be more complex in the case of using multi-gpu. 
    print("Let's use multi-gpu!")
    encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
    decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])


# loss function and optimizer
if subject_id==0:
    loss_fn = nn.L1Loss()
elif subject_id==1:
    loss_fn = nn.CrossEntropyLoss()
elif subject_id==3 or subject_id==4:
    loss_fn_recon = nn.L1Loss()
    loss_fn_class = nn.CrossEntropyLoss()
    
optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-3)
if subject_id!=3:
    # need the optimizer for decoder!
    if subject_id==4:
        optimizer_decoder1 = torch.optim.SGD(decoder.decoder1.parameters(), lr=1e-3)
        optimizer_decoder2 = torch.optim.SGD(decoder.decoder2.parameters(), lr=1e-3)
    else:
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=1e-3)
        
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

log ={"loss":[], "loss_recon":[], "loss_class":[]}

#alternately = True      # determine to train two decoders alternately/simultaneously
alternately_count = 0
alternately_steps = 1

for i, batch in enumerate(tqdm(dataloader)):

    frame_batch = batch[0].to(device)   # (batch_size, C, num_frames, H, W)
    label_batch = batch[1].to(device)   # (batch_size)
        
    features = encoder(frame_batch)     #(batch_size, C_feature, num_frames, H, W)
    if subject_id!=4:
        output = decoder(features)
    elif  subject_id==4:
        # train decoders alternately
        decoder_id = ((alternately_count//alternately_steps) % 2) + 1
        output = decoder(features, decoder_id)
    elif subject_id==5:
        # train decoders simultaneously
        output = decoder(features, 3)

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
        loss = loss_recon + loss_class
        
        # normalization before sum
        
    elif subject_id == 4:
        if decoder_id == 1:
            # used decoder1, so loss function is for reconstruction
            loss = loss_fn_recon(output, frame_batch)
        elif decoder_id == 2:
            # used decoder2, so loss function is for classification
            loss = loss_fn_class(output, label_batch)
        alternately_count += 1
    elif subject_id == 5:
        # loss_recon + loss_class
        loss_recon = loss_fn_recon(output[0], frame_batch)
        loss_class = loss_fn_class(output[1], label_batch)
        loss = loss_recon + loss_class
        
        
    # backpropagation
    optimizer_encoder.zero_grad()
    if subject_id!=3 and subject_id!=4:
        optimizer_decoder.zero_grad()
    elif subject_id==4:
        # alternately
        if decoder_id == 1:
            optimizer_decoder1.zero_grad()
        elif decoder_id == 2:
            optimizer_decoder2.zero_grad()
    elif subject_id==5:
        # simultaneously
        optimizer_decoder1.zero_grad()
        optimizer_decoder2.zero_grad()

            
    #optimizer.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    if subject_id!=3 and subject_id!=4:
        optimizer_decoder.step()
    elif subject_id==4:
        # alternately
        if decoder_id == 1:
            optimizer_decoder1.step()
        elif decoder_id == 2:
            optimizer_decoder2.step()
    elif subject_id==5:
        # simultaneously
        optimizer_decoder1.step()
        optimizer_decoder2.step()            
    #optimizer.step()

    # clear cash
    del frame_batch
    del label_batch
    del features
    del output
    torch.cuda.empty_cache()
    
    if i % 100 == 0:
        if subject_id!=3 and subject_id!=4:
            log["loss"].append(float(loss))
        elif subject_id==3:
            log["loss_recon"].append(float(loss_recon))
            log["loss_class"].append(float(loss_class))
        elif subject_id==4:
            # alternately
            if decoder_id == 1:
                log["loss_recon"].append(float(loss))
            elif decoder_id == 2:
                log["loss_class"].append(float(loss))
        elif subject_id == 5:
            # simultaneously
            log["loss_recon"].append(float(loss_recon))
            log["loss_class"].append(float(loss_class))
                

folder_name = folder_name_list[base_model_id]
input_shape = str(input_H)+'*'+str(input_W)
folder_path = 'result/' + folder_name
os.makedirs(folder_path, exist_ok=True)
torch.save(encoder, folder_path +'/'+ subjects[subject_id]+'_encoder_'+input_shape+'.pth')
torch.save(decoder, folder_path +'/'+ subjects[subject_id]+'_decoder_'+input_shape+'.pth')
with open(folder_path +'/'+ subjects[subject_id]+'_train-history_'+input_shape+'.pkl', 'wb') as f:
    pickle.dump(log, f)

#torch.save(model, 'model_encoder_decoder.pth')

x = range(len(log["loss"]))
plt.plot(x, log["loss"])
plt.show()