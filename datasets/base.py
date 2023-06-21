import imageio.v2
import torch
from torch.utils.data import Dataset
import numpy as np
from .img_color_augmentation import color_transformation_single
from tqdm import tqdm

import os
import cv2
std=0.1

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0,**kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.rgb_chroma_codes=None

        self.rgb_jitter_flag=kwargs.get('jitter_view_appearance',False)
        self.already_jittered=False
        if not self.rgb_jitter_flag:
            self.std=0
        elif kwargs.get('chroma_std',None) is not None:
            std__=kwargs.get('chroma_std',None)
            assert isinstance(std__,float)
            self.std=std__
        else:
            self.std=std
        print(f'base dataset class, kwargs={kwargs},chroma std={self.std}')

    def configure_rgb_chroma_codes(self):
        assert isinstance(self.poses,(torch.Tensor,np.ndarray))
        assert len(self.poses.shape)==3 # N_views, 3, 4
        N_views=self.poses.shape[0]

        file_=os.path.join(self.root_dir,f'chroma_codes_{self.std:.6}.pt')

        generate_chroma_codes=True
        if os.path.exists(file_):
            useful_data=torch.load(file_)
            if self.std != useful_data.get('std'):
                print('different chroma jitter configuration, regenerate')
                generate_chroma_codes=True
            else:
                print('same chroma jitter configuration, do not regenerate')
                generate_chroma_codes=False
                self.rgb_chroma_codes=useful_data.get('chroma_codes')
        else:
            print('chroma jitter configuration not found, regenerate')


        if generate_chroma_codes:
            self.rgb_chroma_codes=np.random.normal(0,self.std,size=(N_views,3))
            self.rgb_chroma_codes[0,:]=0 # assume the first pose camera is correct

            self.chroma_configuration_={
                'chroma_codes':self.rgb_chroma_codes,
                'std':self.std

            }
            torch.save(self.chroma_configuration_,file_)





    def rgb_chroma_transform_all(self):
        if self.rgb_chroma_codes is None:
            self.configure_rgb_chroma_codes()
        if not self.split.startswith('train'):
            return
        assert not self.already_jittered
        assert isinstance(self.rays,torch.Tensor)
        N_views=self.poses.shape[0]

        print(f'color jitter the dataset, split={self.split}')
        #print(f'self.rgb_chroma_codes={self.rgb_chroma_codes}')
        img_diff=np.zeros(N_views,dtype=np.float64)
        if self.std>0:
            device_=self.rays.device
            self.rays=self.rays.to('cpu')
            for i in tqdm(range(N_views)):
                img=self.rays[i].view(self.img_wh[1],self.img_wh[0], 3).numpy()
                img=img*255
                img=img.astype(np.uint8)
                img_transformed=color_transformation_single(uint8_img=img,Hue_offset=self.rgb_chroma_codes[i,0],
                                                            gamma_S=self.rgb_chroma_codes[i,1]/4,


                                                            gamma_V=self.rgb_chroma_codes[i,2]/4,

                                                            )
                img_diff[i]= np.mean(np.abs((img.astype(float)-img_transformed.astype(float))))
                try:
                    img_name_tmp=f"{self.frames[i]['file_path']}_jitter_{self.std:.3}.png"
                    img_jittered_save=os.path.join(self.root_dir,img_name_tmp)

                except:
                    # assuming running in linux
                    # https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
                    img_name_tmp=(f"{os.path.basename(self.img_paths[i])[:-4]}_jitter_{self.std:.3}.png")

                    img_jittered_save=os.path.join(self.root_dir,'rgb',img_name_tmp)
                imageio.v2.imsave(img_jittered_save,img_transformed)
                img_t_tensor=torch.from_numpy(img_transformed/255)
                img_t_tensor=img_t_tensor.view(self.img_wh[0]*self.img_wh[1],3)
                if np.linalg.norm(self.rgb_chroma_codes[i,:])>0 :
                    assert torch.norm(self.rays[i] - img_t_tensor) > 0
                self.rays[i]=img_t_tensor
            self.rays = self.rays.to(device_)
            print(f'img_diff={img_diff},{img_diff.mean()}')
            self.img_diff_mean=img_diff.mean()/255

        self.already_jittered=True


    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
                img_idxs = np.full(self.batch_size,fill_value=img_idxs).astype(int)
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
#            print(f'test')
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample