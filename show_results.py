from torchvision import transforms
import matplotlib.pyplot as plt

#########################################################################
# for reshape to check result by image
tfs_re = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(1, 2, 3, 0)),
            #transforms.Lambda(lambda x: x * 255),
])

def plot_image(x):
    #x = x[0][0][0]             # get 1 frame
    x = normalize(x)
    plt.figure()
    plt.imshow(x.cpu())

def normalize(x):
    # x: (H, W, C)
    max = x.max()
    min = x.min()
    x = (x-min)/(max-min) 
    return x
    
########################################################################

def plot_images(dataloader, encoder, decoder, device):
    itr_dataloader = iter(dataloader)
    x = next(itr_dataloader)
    frame_batches = x[0]
    reshaped_batch = tfs_re(frame_batches[0])   
    plot_image(reshaped_batch[0])
    output = encoder(frame_batches.to(device))
    output = decoder(output)
    output = tfs_re(output[0])
    plot_image(output[0].detach())