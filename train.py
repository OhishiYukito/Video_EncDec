import models
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101
from tqdm import tqdm
import matplotlib.pyplot as plt

# create Dataset object and Dataloader
# The path to root directory, which contains UCF101 video files (not rawframes)
root_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucf101/UCF-101'

ann_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucfTrainTestSplit'

batch_size = 8

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
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
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
#if device == "cuda:0":
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
    
#    for i in range(batch_size):
#        # get one video frames and its label
#        frames = frame_batch[i]
#        label = label_batch[i]
        
#        # encode and decode
#        feature = encoder(frames)
#        output = decoder(label)
        
#        # use loss function and labels to evaluate
#        loss = loss_fn(output, frames)
        
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
    #del features
    del output
    torch.cuda.empty_cache()
    
    if i % 100 == 0:
        log["loss"].append(float(loss))

    
torch.save(encoder, 'result/model_encoder.pth')
torch.save(decoder, 'result/model_decoder.pth')
#torch.save(model, 'model_encoder_decoder.pth')

x = range(len(log["loss"]))
plt.plot(x, log["loss"])
plt.show()