from numpy import number, reshape
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from tqdm import tqdm
import os

#########################################################################
# for reshape to check result by image
tfs_re = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(1, 2, 3, 0)),
            #transforms.Lambda(lambda x: x * 255),
])

# plot an image
def plot_image(x, fig=None):
    #x = x[0][0][0]             # get 1 frame
    x = normalize(x)
    if fig:
        fig.imshow(x.cpu())
    else:
        plt.figure()
        plt.imshow(x.cpu())

def normalize(x):
    # x: (H, W, C)
    max = x.max()
    min = x.min()
    x = (x-min)/(max-min) 
    return x
    
def topk_research(x, true_indicies, k=1):
    topk = torch.topk(x, k, dim=1)
    #topk_value = [value for value in topk.values]
    topk_indices = topk.indices.T
    total = x.eq(topk_indices).reshape(-1).float().sum(0, keepdim=True)
    return total
########################################################################

def plot_reconstruction(dataloader, encoder, decoder, device):
    number_of_indication = 0    
    n = 3
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module
        
    for j, x in enumerate(tqdm(dataloader)):
        # indicate only n samples
        if number_of_indication >= n:
            break

        if j%100==0:
            fig = plt.figure()
            frame_batches = x[0]
            reshaped_frames = tfs_re(frame_batches[0])
            # plot an origin image
            ax1 = fig.add_subplot(1, 2, 1)
            plot_image(reshaped_frames[0], ax1)
            
            output = encoder(frame_batches.to(device))
            output = decoder(output)
            output = tfs_re(output[0])
            # plot a model's output
            ax2 = fig.add_subplot(1, 2, 2)
            plot_image(output[0].detach(), ax2)
            #plt.show()
            number_of_indication += 1
        else:
            continue
        
    plt.show()


def plot_classification(dataloader, encoder, decoder, device):
    d = {}
    with open("dataset/ucfTrainTestSplit/classInd.txt") as f:
        for line in f:
            (key, val) = line.split()
            d[int(key)] = val

    number_of_indication = 0    
    n = 3
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module
        
    for j, x in enumerate(tqdm(dataloader)):
        # indicate only n samples
        if number_of_indication >= n:
            break

        if j%100==0:
            frame_batches = x[0]
            label_batches = x[1]
            reshaped_frames = tfs_re(frame_batches[0])
            # plot an origin image
            plot_image(reshaped_frames[0])
            
            output = encoder(frame_batches.to(device))
            output = decoder(output)
            output = torch.nn.Softmax(dim=1)(output)

            top3 = torch.topk(output[0], 3)
            top3_value = [value for value in top3.values]
            top3_indices = top3.indices
            top3_class = [d[int(i)] for i in top3_indices]
            text = "true_label : " + d[int(label_batches[0])+1]+"\n"
            text += top3_class[0]+" : "+str(round(float(top3_value[0]), 3))+"\n"
            text += top3_class[1]+" : "+str(round(float(top3_value[1]), 3))+"\n"
            text += top3_class[2]+" : "+str(round(float(top3_value[2]), 3))
            # add top 3 scores as text
            plt.text(1,0, text)
            #plt.show()
            number_of_indication += 1
        else:
            continue
            
    plt.show()


def plot_recon_class(dataloader, encoder, decoder, device, subject_id):
    d = {}
    with open("dataset/ucfTrainTestSplit/classInd.txt") as f:
        for line in f:
            (key, val) = line.split()
            d[int(key)] = val

    number_of_indication = 0    
    n = 3
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module

    for j, x in enumerate(tqdm(dataloader)):
        # indicate only n samples
        if number_of_indication >= n:
            break

        if j%100==0:
            fig = plt.figure()
            frame_batches = x[0]
            label_batches = x[1]
            reshaped_frames = tfs_re(frame_batches[0])
            # plot an origin image
            ax1 = fig.add_subplot(1, 2, 1)
            plot_image(reshaped_frames[0], ax1)
            
            output = encoder(frame_batches.to(device))
            if subject_id == 3:
                output = decoder(output)
            elif subject_id == 4:
                output = decoder(output, 3)

            output_prob = torch.nn.Softmax(dim=1)(output[1])
            top3 = torch.topk(output_prob[0], 3)
            top3_value = [value for value in top3.values]
            top3_indices = top3.indices
            top3_class = [d[int(i)] for i in top3_indices]
            text = "true_label : " + d[int(label_batches[0])+1]+"\n"
            text += top3_class[0]+" : "+str(round(float(top3_value[0]), 3))+"\n"
            text += top3_class[1]+" : "+str(round(float(top3_value[1]), 3))+"\n"
            text += top3_class[2]+" : "+str(round(float(top3_value[2]), 3))
            # add top 3 scores as text
            plt.text(1,0, text)
            

            output_frames = tfs_re(output[0][0])
            # plot a model's output
            ax2 = fig.add_subplot(1, 2, 2)
            plot_image(output_frames[0].detach(), ax2)
            #plt.show()

            
            number_of_indication += 1
        else:
            continue
    plt.show()


def plot_animation(dataloader, encoder, decoder, device, base_model_name, subject_name):
    number_of_indication = 0    
    n = 3
    interval = 100
    if getattr(encoder, 'device_ids', False):
        encoder = encoder.module
        decoder = decoder.module
    origin_imgs = []
    output_imgs = []    


    for j, x in enumerate(tqdm(dataloader)):
        # indicate only n samples
        if number_of_indication >= n:
            break

        if j%interval==0:
            mix_imgs = []
            fig = plt.figure()
            frame_batches = x[0]
            reshaped_frames = tfs_re(frame_batches[0])
            
            output = encoder(frame_batches.to(device))
            output = decoder(output)
            output = tfs_re(output[0])
            
            
            # plot an origin image
            # plot a model's output
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            for i in range(reshaped_frames.shape[0]):
                ax = plt.subplot(1, 2, 1)
                origin_im = normalize(reshaped_frames[i])
                origin_im = ax.imshow(origin_im.cpu())
                #origin_imgs.append([origin_im])

                ax = plt.subplot(1, 2, 2)
                output_im = normalize(output[i].detach())
                output_im = ax.imshow(output_im.cpu())
                #output_imgs.append([output_im])
                mix = [origin_im]
                mix.append(output_im)
                mix_imgs.append(mix)
                #mix_imgs.append([output_im])
            #plot_image(output[0].detach(), ax2)
            #plt.show()
            number_of_indication += 1
            
            ani_mix = animation.ArtistAnimation(fig, mix_imgs, interval=1000, blit=True, repeat_delay=1000)
            os.makedirs("output/"+base_model_name, exist_ok=True)
            ani_mix.save("output/"+base_model_name+"/" + base_model_name+"--"+subject_name +"_"+str(j//interval) + ".gif")
        else:
            continue
    
    #ani_origin = animation.ArtistAnimation(fig, origin_imgs, interval=100, blit=True, repeat_delay=1000)
    #ani_output = animation.ArtistAnimation(fig, output_imgs, interval=100, blit=True, repeat_delay=1000)
    plt.show()
    
    gif_origin_path = "output/" + subject_name + "_origin.gif"
    gif_output_path = "output/" + subject_name + "_output.gif"
    
    #ani_origin.save(gif_origin_path, writer="ffmpeg")
    #ani_output.save(gif_output_path, writer="ffmpeg")
    