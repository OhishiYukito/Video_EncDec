import models
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# create Dataset object and Dataloader
# The path to root directory, which contains UCF101 video files (not rawframes)
input_H = 120
input_W = 160
batch_size = 8
lab_server_pc = True

if lab_server_pc:
    root_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucf101/UCF-101'
    ann_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucfTrainTestSplit'
else:
    root_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/videos'
    ann_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/ucfTrainTestSplit'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                    train=True,
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

# create model instance
encoder = models.Encoder(3)
decoder = models.DecoderToFrames(3)
#model = models.EncoderDecoder(3)
 
encoder.train()
decoder.train()
#model.train()

encoder.to(device)
decoder.to(device)
#model.to(device)

# for using multi-gpu
if lab_server_pc:
    print("Let's use multi-gpu!")
    encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3])
    decoder = nn.DataParallel(decoder, device_ids=[0,1,2,3])
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])


# loss function and optimizer
loss_fn = nn.L1Loss()
optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=1e-3)
optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

log ={"loss":[]}


for i, batch in enumerate(tqdm(dataloader)):

    frame_batch = batch[0].to(device)
    #label_batch = batch[1].to(device)
        
    features = encoder(frame_batch)
    output = decoder(features)
    #output = model(frame_batch)
    loss = loss_fn(output, frame_batch)
    
    # backpropagation
    optimizer_decoder.zero_grad()
    optimizer_encoder.zero_grad()
    #optimizer.zero_grad()
    loss.backward()
    optimizer_decoder.step()
    optimizer_encoder.step()
    #optimizer.step()

    # clear cash
    del frame_batch
    #del label_batch
    del features
    del output
    torch.cuda.empty_cache()
    
    if i % 100 == 0:
        log["loss"].append(float(loss))


folder_name = 'result/'+str(input_H)+'*'+str(input_W)
os.makedirs(folder_name, exist_ok=True)
torch.save(encoder, folder_name+'/model_encoder_'+folder_name+'.pth')
torch.save(decoder, folder_name+'/model_decoder_'+folder_name+'.pth')
#torch.save(model, 'model_encoder_decoder.pth')

x = range(len(log["loss"]))
plt.plot(x, log["loss"])
plt.show()