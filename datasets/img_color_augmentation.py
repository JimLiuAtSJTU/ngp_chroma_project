import copy

import cv2
import numpy as np


def patch_transformation(input_img: np.ndarray, patch_size: tuple, dh: np.ndarray, ds: np.ndarray, dv: np.ndarray):

    img_output=copy.deepcopy(input_img)

    assert input_img.shape[2]==3 # RGB image


    chroma_codes=np.zeros_like(input_img,dtype=float)



    m,n=(patch_size)


    assert dh.shape == patch_size == ds.shape == dv.shape


    xdim,ydim=input_img.shape[0],input_img.shape[1]

    stride_x =  round(xdim/m)
    stride_y = round(ydim/n)


    for i in range(m):
        for j in range(n):

            end_x= (i+1)*stride_x if i+1<m else (xdim)
            end_y= (j+1)*stride_y if j+1<n else ydim


            patch=input_img[i*stride_x:end_x, j*stride_y:end_y ]

            img_output[i*stride_x:end_x, j*stride_y:end_y ] = color_transformation_single(
                uint8_img=patch, Hue_offset=dh[i,j] , gamma_S= ds[i,j], gamma_V= dv[i,j]
            )

            code_=[dh[i,j],ds[i,j],dv[i,j]]
            hsv_code=np.array(code_,dtype=float)

            # set the chroma codes of each pixel
            chroma_codes[i*stride_x:end_x, j*stride_y:end_y] = hsv_code












    return img_output,chroma_codes




def patchify_image(input_img:np.ndarray, patch_size):


    if isinstance(patch_size,int):
        m=patch_size
        n=patch_size
    else:
        assert isinstance(patch_size,list)\
               or isinstance(patch_size,tuple)\
               or ( isinstance(patch_size,np.ndarray)
                    and len(patch_size)==1
                    and patch_size.shape[0]==2)

        m,n=tuple(patch_size)

    xdim,ydim=input_img.shape[0],input_img.shape[1]

    stride_x = np.round(xdim/m)
    stride_y = np.round(ydim/n)

    patches={}
    for i in range(m):
        for j in range(n):

            end_x= (i+1)*stride_x if i+1<m else (xdim)
            end_y= (j+1)*stride_y if j+1<n else ydim


            patch=input_img[i*stride_x:end_x, j*stride_y:end_y         ]

            patches[f'{i},{j}']=patch

    return patches


def unpatchify_image(input_shape,m,n,patches:dict):

    merged=np.zeros(input_shape)

    xdim, ydim = input_shape[0], input_shape[1]

    stride_x = np.round(xdim / m)
    stride_y = np.round(ydim / n)

    for i in range(m):
        for j in range(n):
            end_x = (i + 1) * stride_x if i + 1 < m else (xdim)
            end_y = (j + 1) * stride_y if j + 1 < n else ydim
            patch_=patches[f'{i},{j}']

            merged[i * stride_x:end_x, j * stride_y:end_y]=patch_


    return merged


def gamma_correction(img, c=1, g=1.15):
    out = img.copy()
    out = out / 255.
    out = (1/c * out) ** (1/g)

    out *= 255
    out = out.astype(np.uint8)

    return out



def float_to_uint8_img(float_img:np.ndarray):
    assert (0<=float_img).all() and (1>=float_img).all()
    re_scaled=float_img*255

    return re_scaled.astype('uint8')

def uint8_img2float(uint8img:np.ndarray):

    return uint8img.astype(float) / 255.0


def color_transformation_multi(uint8_image_set:np.ndarray, Hue_offset_s:np.ndarray, gamma_s:np.ndarray,gamma_v:np.ndarray):
    assert len(uint8_image_set.shape) == 4 # N * h* w * channels
    dataset_length= uint8_image_set.shape[0]
    assert dataset_length==Hue_offset_s.shape[0]
    assert dataset_length==gamma_s.shape[0]

    result_=uint8_image_set.astype('uint8').copy()


    for i in range(dataset_length):
        uint8_img=uint8_image_set[i]
        result_[i]=color_transformation_single(uint8_img,Hue_offset=Hue_offset_s[i],gamma_S=gamma_s[i],gamma_V=gamma_v[i])

    return result_



def color_transformation_single(uint8_img:np.ndarray, Hue_offset:float, gamma_S:float,gamma_V:float):


    '''
    Hue offset: [ -0.5 , 0.5]
    setting hue offset = 0, gamma_S = 0, gamma_V =0, indicates no transformation, i.e. identical images.


    img: SHOULD BE RGB image
    Watch out for OpenCV BGR/ RGB channel problems.
    '''
    #assert Hue_offset>=-0.5 and Hue_offset<=0.5

    Hue_offset=np.clip(Hue_offset,-0.5,0.5)
    gamma_S=np.clip(gamma_S,-0.5,0.5)
    gamma_S=np.clip(gamma_S,-0.5,0.5)



    assert len(uint8_img.shape)==3 and uint8_img.shape[2]==3 # caution. must be RGB image.
    assert isinstance(uint8_img[0,0,0],np.uint8)
    img_in_hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
    H_channel = img_in_hsv[:, :, 0]

    offset_=round( ( Hue_offset) *180)  # [ - 90, 90 ]

    '''
    uniform distribution.
    '''
    H_channel_transformed = (H_channel + offset_ + 180  ) % 180

    img_in_hsv[:,:,0]=H_channel_transformed


    S_channel = img_in_hsv[:, :, 1]

    S_channel_transformed = gamma_correction(S_channel,g=gamma_S+1)

    img_in_hsv[:,:,1]=S_channel_transformed

    V_channel = img_in_hsv[:, :, 2]
    V_channel_transformed = gamma_correction(V_channel,g=gamma_V+1)
    img_in_hsv[:,:,2]=V_channel_transformed

    transformed_img=cv2.cvtColor(img_in_hsv, cv2.COLOR_HSV2RGB)

    return (transformed_img)

