import torch
from torchvision.transforms import functional
from torchvision import transforms
import cv2
import numpy as np
import numbers
import PIL
import torch

from timm.data.random_erasing import RandomErasing
from runner.src.dataset import basic_transforms
from runner.src.dataset.basic_transforms import RandomVerticalFlip, RandomHorizontalFlip, pad_frames
from pytorchvideo import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CENTER_MEAN = [0.5, 0.5, 0.5]
CENTER_STD = [0.5, 0.5, 0.5]

OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD= [0.26862954, 0.26130258, 0.27577711]

BASIC_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    #"Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
]

def gt_cls_target(curtime_batch, toa_batch, tea_batch):
    return (
        (toa_batch >= 0) &
        (curtime_batch >= toa_batch) & (
            (curtime_batch < tea_batch) |
            # case when sub batch end with a positive frame
            (toa_batch == tea_batch)
        )
    )

def plain_transforms(crop_h, crop_w, mean_std, **args):
    if mean_std == 'center':
        MEAN, STD = IMAGENET_MEAN, CENTER_STD
    else:
        MEAN, STD = CENTER_MEAN, CENTER_STD
    return transforms.Compose([
            transforms.Lambda(lambda x: np.array([
                cv2.resize(img, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                for img in x
            ]).astype(np.float32) / 255.0), 
            transforms.Lambda(lambda x: torch.from_numpy(x)), 
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  
            transforms.Normalize(MEAN, STD),
        ])
        
def movad_transforms(crop_h, crop_w, mean_std, **args):
    
    if mean_std == 'center':
        MEAN, STD = IMAGENET_MEAN, CENTER_STD
    else:
        MEAN, STD = CENTER_MEAN, CENTER_STD
    
    is_augmix = args.get('augmix', False)
    
    return transforms.Compose([
                pad_frames([crop_h, crop_w]),
                transforms.Lambda(lambda x: torch.tensor(x)),
                # [T, H, W, C] -> [T, C, H, W]
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                T.AugMix() if is_augmix else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(MEAN, STD),
            ])

class PlainAugmentor:
    def __init__(self, crop_h, crop_w, origin_shape, **args):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.origin_shape = origin_shape 
        self.args = args
        self.mean_std = self.args.get('mean_std','center')
        self.MEAN , self.STD = (IMAGENET_MEAN, CENTER_STD) if self.mean_std == 'imagenet' else (CENTER_MEAN, CENTER_STD)
        self.vertical_flip_prob = args.get('vertical_flip_prob', 0.0)
        self.horizontal_flip_prob = args.get('horizontal_flip_prob', 0.0)
        self.is_augmix = args.get('augmix', False)
        self.is_padding = args.get('is_padding', False)
        # resize 函数：直接resize或者保持原图比例padding黑边
        if self.is_padding:
            resize_func = pad_frames([crop_h, crop_w])  
        else:
            resize_func = transforms.Lambda(lambda x: np.array([
                cv2.resize(img, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR) for img in x ]))
               
        self.transform = transforms.Compose([
                resize_func,
                transforms.Lambda(lambda x: torch.tensor(x)),
                # [T, H, W, C] -> [T, C, H, W]
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                T.AugMix() if self.is_augmix else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(self.MEAN , self.STD),
            ])
        if self.vertical_flip_prob > 0.:
            self.vflipper = RandomVerticalFlip(self.origin_shape, self.vertical_flip_prob)

        if self.horizontal_flip_prob > 0.:
            self.hflipper = RandomHorizontalFlip(self.origin_shape, self.horizontal_flip_prob)

    def __call__(self, buffer , boxes):
        """
        buffer: list[np.ndarray] or list[Tensor]  (T frames, shape: H W C)
        return: Tensor (C, T, H, W)
        """
        buffer = self.transform(buffer)

        if hasattr(self, 'hflipper'):
            _, buffer, boxes = self.hflipper(buffer, boxes)

        if hasattr(self, 'vflipper'):
            _, buffer, boxes = self.vflipper(buffer, boxes)

        return buffer, boxes



class RandAugmentor:
    def __init__(self, crop_h, crop_w, origin_shape, rand_erase, **args):
        """
        Args:
            crop_h, crop_w (int): 最终裁剪大小
            rand_erase (bool): 是否使用 RandErasing
            args: 包含 aa, train_interpolation, reprob, remode, recount 等字段
        """
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.rand_erase = rand_erase
        self.origin_shape = origin_shape 
        self.args = args
        self.mean_std = self.args.get('mean_std','imagenet')
        self.MEAN , self.STD = (IMAGENET_MEAN, CENTER_STD) if self.mean_std == 'imagenet' else (CENTER_MEAN, CENTER_STD)
        self.trasnforms = BASIC_TRANSFORMS
        self.aug_transform = basic_transforms.create_random_augment(
            input_size=(crop_h, crop_w),
            auto_augment=args['aa'],
            interpolation=args['train_interpolation'],
            do_transforms=self.trasnforms
        )
        self.vertical_flip_prob = args.get('vertical_flip_prob', 0.0)
        self.horizontal_flip_prob = args.get('horizontal_flip_prob', 0.0)
        if self.vertical_flip_prob > 0.:
            self.vflipper = RandomVerticalFlip(self.origin_shape, self.vertical_flip_prob)

        if self.horizontal_flip_prob > 0.:
            self.hflipper = RandomHorizontalFlip(self.origin_shape, self.horizontal_flip_prob)

        # Rand Erasing
        if rand_erase:
            assert 'erase_cfg' in args, "rand_erase is True but erase_cfg not in args"
            self.erase_cfg = args['erase_cfg']
            self.erase_transform = RandomErasing(
                self.erase_cfg.reprob,
                mode=self.erase_cfg.remode,
                max_count=self.erase_cfg.recount,
                num_splits=self.erase_cfg.recount,
                max_area=0.1,
                device="cpu",
            )
    
    @staticmethod
    def tensor_normalize(tensor, mean, std):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize. [B, C, H, W] or [C, H, W]
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        functional.normalize(tensor, mean, std)
        return tensor

    def __call__(self, buffer , boxes):
        """
        buffer: list[np.ndarray] or list[Tensor]  (T frames, shape: H W C)
        return: Tensor (C, T, H, W)
        """
        h, w, _ = buffer[0].shape

        # 1. padding
        do_pad = basic_transforms.pad_wide_clips(h, w, self.crop_h, self.crop_w)
        buffer = [do_pad(img) for img in buffer]

        # 2. RandAugment
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = self.aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]

        # stack -> T,C,H,W
        buffer = torch.stack(buffer)

        # 3. Normalize
        buffer = self.tensor_normalize(buffer, self.MEAN, self.STD)

        # 4. Rand Erasing
        if self.rand_erase:
            buffer = self.erase_transform(buffer)

        # 5. flip with boxes
        if hasattr(self, 'hflipper'):
            _, buffer, boxes = self.hflipper(buffer, boxes)

        if hasattr(self, 'vflipper'):
            _, buffer, boxes = self.vflipper(buffer, boxes)

        return buffer, boxes

