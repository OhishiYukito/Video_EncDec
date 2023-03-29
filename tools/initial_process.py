import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import UCF101


def initial_process(lab_server_pc, subject_id, input_H, input_W, batch_size, train=True):
    # create Dataset object and Dataloader
    # The path to root directory, which contains UCF101 video files (not rawframes)
    if lab_server_pc:
        root_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucf101/UCF-101'
        ann_dir = '/home/all/Desktop/Ohishi/Video_EncDec/dataset/ucfTrainTestSplit'
    else:
        root_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/videos'
        ann_dir = '/home/ohishiyukito/Documents/GraduationResearch/data/ucf101/ucfTrainTestSplit'

    class_indxs = {}
    if subject_id == 1 or subject_id == 4:
        # class_index dictionary
        with open("dataset/ucfTrainTestSplit/classInd.txt") as f:
            for line in f:
                (key, val) = line.split()
                class_indxs[int(key)-1] = val
                

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
                    train=train,
                    transform=tfs,
                    num_workers=20)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=train,  # if train_mode, dataset will be shuffled, but in test_mode, shuffle isn't required
                                            collate_fn=custom_collate, 
                                            num_workers=1)
    # data : ((batch_size, C, num_frames, H, W), (class_ids))
    # data[1] has batch_size individual class_id.
    # example : if batch_size=4, len(data[1])=4

    return class_indxs, device, dataloader