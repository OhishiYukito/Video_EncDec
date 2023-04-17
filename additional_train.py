import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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


base_model_id = 2
folder_name_list = {
    0 : "Conv3d",       # if base_model is Conv3d
    1 : "(2+1)D_without_softmax",       # if base_model is Conv2Plus1D
    2 : "(2+1)D",
}

load_model_epoch = 1
additional_epoch = 4
##########################################################

class_indxs, device, dataloader = init.initial_process(lab_server_pc,
                                                       subject_id, 
                                                       input_H, 
                                                       input_W, 
                                                       batch_size, 
                                                       train=True)

print(folder_name_list[base_model_id], subjects[subject_id], input_H, input_W)


# create model instance
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
with open(file_path_incomplete +'_train-history_'+ file_path_last_part+'.pkl', 'rb') as f:
    log = pickle.load(f)

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


log_interval = 100
#alternately = True      # determine to train two decoders alternately/simultaneously
alternately_steps = 1
decoder_id = 2          # in the first iteration, decoder_id will be changed,
                        # so set decoder_id=2 to start with decoder_id=1
record_both = True      # to check whether two decoder's log was recorded

for epoch in range(additional_epoch):
    print("epoch: ", load_model_epoch+epoch+1)
    for key in log:
        log[key] = log[key].tolist()

    for i, batch in enumerate(tqdm(dataloader)):

        frame_batch = batch[0].to(device)   # (batch_size, C, num_frames, H, W)
        label_batch = batch[1].to(device)   # (batch_size)
            
        features = encoder(frame_batch)     #(batch_size, C_feature, num_frames, H, W)
        if subject_id!=4:
            output = decoder(features)
        elif  subject_id==4:
            # train decoders alternately
            if i % alternately_steps ==0:
                decoder_id = (decoder_id % 2) +1
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
        

        if subject_id==4:
            if i%log_interval == 0 and record_both:
                # record log only one side
                # alternately
                if decoder_id == 1:
                    log["loss_recon"].append( [i, float(loss)] )
                    recorded_id = 1
                elif decoder_id == 2:
                    log["loss_class"].append( [i, float(loss)] )
                    recorded_id = 2
                record_both = False
            elif not record_both and decoder_id!=recorded_id:
                # record other side
                if decoder_id == 1:
                    log["loss_recon"].append( [i, float(loss)] )
                elif decoder_id == 2:
                    log["loss_class"].append( [i, float(loss)] )
                record_both = True

        elif i%log_interval==0:
            if subject_id!=3:
                log["loss"].append( [i, float(loss)] )
            elif subject_id==3:
                log["loss_recon"].append( [i, float(loss_recon)] )
                log["loss_class"].append( [i, float(loss_class)] )
            elif subject_id == 5:
                # simultaneously
                log["loss_recon"].append( [i, float(loss_recon)] )
                log["loss_class"].append( [i, float(loss_class)] )
                    
    for key in log:
        log[key] = np.array(log[key])

    trained_epochs = load_model_epoch + epoch + 1
    torch.save(encoder, file_path_incomplete+'_encoder_'+input_shape+'_'+str(trained_epochs).zfill(2)+'epoch'+'.pth')
    torch.save(decoder, file_path_incomplete+'_decoder_'+input_shape+'_'+str(trained_epochs).zfill(2)+'epoch'+'.pth')
    with open(file_path_incomplete+'_train-history_'+input_shape+'_'+str(trained_epochs).zfill(2)+'epoch'+'.pkl', 'wb') as f:
        pickle.dump(log, f)


if subject_id!=4:
    x = log["loss"][:, 0]
    y = log["loss"][:, 1]
    plt.plot(x, y)
else:
    x1 = log["loss_recon"][:, 0]
    y1 = log["loss_recon"][:, 1]
    x2 = log["loss_class"][:, 0]
    y2 = log["loss_class"][:, 1]
    plt.figure()
    plt.title("loss_recon")
    plt.plot(x1, y1)

    plt.figure()
    plt.title("loss_class")
    plt.plot(x2, y2)

plt.show()