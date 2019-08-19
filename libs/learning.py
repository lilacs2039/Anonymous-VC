#!/usr/bin/env python
"learning and plotting."


# import librosa
# import pyworld as pw
# import scipy
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import IPython.display

import fastai as fastai
from fastai.vision import *
from fastai.vision.image import Image
from fastai.vision.data import *
from fastai.vision.gan import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.callbacks.tensorboard import *
from torchvision.models import resnet18

from .audio import *

# basical operations
def image_to_np(image:Image)->np.array:
    "returns (Dabs,Dphase)"
    a = (image._px).numpy()  #Image -> Tensor -> numpy
    return a[0],a[1]
def np_to_image(abs:np.array,phase:np.array)->Image:
    a = np.array([abs,phase,np.zeros(abs.shape)])
    a = torch.Tensor(a) #ndarray -> Tensor
    return Image(a)  # Tensor -> Image
def savepickle(p:Path,abs,phase):
    if(p.suffix != '.pkl'):p =p/".pkl"
    with open(p,'wb') as f:
        pickle.dump(np_to_image(abs,phase),f)
def open_from_pickle(fn)->Image:
    """
    fn: path of pickled Image
    Returns:
      The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    with open(fn,"rb") as f:
        a  =pickle.load(f)
        assert type(a)==Image, f"'{fn}' is not Image type"
        return a
def conc_specimage(image:Image):
    """
    concatenate each 3 channels of image vertically, then, returns 3times height 1ch image.
    """
    abs,phase = image_to_np(image)  # Image -> ndarray
    return np.concatenate([abs,phase,np.zeros(abs.shape)],axis=0)


# pre/post process functions
def preprocess_input(input_file:Path,original_output_dir:Path,anonymized_output_dir:Path):
    """
    input audio, outputs original specimage,
    :param original_fullpath:
    :param output_dir:
    :return:
    """
    src ,fs= librosa.load(input_file.resolve(), sr=conf.sample_ratio)
    if  conf.prep_audio_dataset_second is not None:
        src = clip_audio_length(src,fs,second=conf.prep_audio_dataset_second)

    # preprocess src audio
    src_abs, src_phase = convert_to_spectrogram(src)
    src_abs ,src_phase = padding_spectrogram(src_abs), padding_spectrogram(src_phase)
    src_abs,_,_= normalize_abs(src_abs)
    src_phase,_,_= normalize_phase(src_phase)

    # preprocess anonymized audio
    anonymized = anonymization(fs, src)
    anonymized_abs, anonymized_phase = convert_to_spectrogram(anonymized)
    anonymized_abs, anonymized_phase = padding_spectrogram(anonymized_abs), padding_spectrogram(anonymized_phase)
    anonymized_abs,_,_= normalize_abs(anonymized_abs)
    anonymized_phase,_,_= normalize_phase(anonymized_phase)

    with open(original_output_dir/(input_file.name+".pkl"),'wb') as f:
        pickle.dump(np_to_image(src_abs,src_phase),f)
    with open(anonymized_output_dir/(input_file.name+".pkl"),'wb') as f:
        pickle.dump(np_to_image(anonymized_abs,anonymized_phase),f)

    #     #save as png image.
    #     save_spectrogram_asImage(src_abs,src_phase,path_prep_original/(p.name+"_image.png"))
    #     save_spectrogram_asImage(anonymized_abs,anonymized_phase,path_prep_anonymous/(p.name+"_image.png"))

    print("processed {}".format(input_file.name+".pkl"))
    return src_abs, src_phase, anonymized_abs, anonymized_phase

def postprocess(Dabs,Dphase):
    if(type(Dabs)==torch.Tensor): Dabs=Dabs.numpy()
    if(type(Dphase)==torch.Tensor): Dphase=Dphase.numpy()
    Dabs = denormalize_abs(Dabs)
    Dphase = denormalize_phase(Dphase)
    y_hat = convert_to_wave(Dabs,Dphase)
    return y_hat

def postprocess_tensor(a:Tensor):
    return postprocess(a[0],a[1])


# plots, representation of images/audios
def hist(a,title="",savepath=None):
    plt.title(title)
    plt.hist(a, 20,
             weights=None, density=False,
             histtype="step", log=False)
    if savepath!=None: plt.savefig(savepath)
    plt.show()

def display_audio(abs,phase,text=""):
    audio=postprocess(abs,phase)
    print(text)
    IPython.display.display(IPython.display.Audio(
        audio, rate=conf.sample_ratio))
def display_audio_from_image(image:Image,text=""):
    abs,phase = image_to_np(image)  # Image -> ndarray
    return display_audio(abs,phase,text)


# ItemList classes for pickled image
class PklImageList(ImageList):
    def open(self,fn):
        """
        used instead of ImageList#open()
        reference : https://github.com/fastai/fastai/blob/1.0.57/fastai/vision/data.py#L267
        """
        return open_from_pickle(fn)

class PklImageImageList(ImageImageList):
    _label_cls = PklImageList
    def open(self,fn):
        return open_from_pickle(fn)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        """
        Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        refer : https://github.com/fastai/fastai/blob/1.0.57/fastai/vision/data.py#L426
        """
        if("showtext" in kwargs):print(kwargs["showtext"])
        title = 'Input / Prediction / Target'
        axs = subplots(len(xs), 3, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            axs[i,0].imshow(conc_specimage(x).T)  #x.show(ax=axs[i,0], **kwargs)
            axs[i,2].imshow(conc_specimage(y).T)  #y.show(ax=axs[i,2], **kwargs)
            axs[i,1].imshow(conc_specimage(z).T)  #z.show(ax=axs[i,1], **kwargs)
            display_audio_from_image(x,f"input{str(i)}")
            display_audio_from_image(z,f"prediction{str(i)}")
            display_audio_from_image(y,f"target{str(i)}")
        plt.show()

# loss func        
def L1LossFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.L1Loss`, but flattens input and target. nn.L1Loss docs:https://pytorch.org/docs/stable/nn.html#l1loss"
    return FlattenedLoss(nn.L1Loss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

