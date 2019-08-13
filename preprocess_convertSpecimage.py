import glob

import util.util_voicepix2pix as util
import librosa
import numpy as np
import pyworld as pw
import scipy
import os
import numpy as np
import argparse
from PIL import Image
import scipy.misc


parser = argparse.ArgumentParser('preprocess audioData')
parser.add_argument('--audio_path', dest='audio_path', help='input directory for audio data', type=str)
parser.add_argument('--output_path', dest='output_path', help='output directory for spectrogram image data', type=str)
parser.add_argument('--audio_dataset_second', dest='audio_dataset_second', help='if not None, clip audio time length to specified seconds.', type=int,default=None)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

print("search dataset paths")
print("    from: %s"%args.audio_path)
wav_paths = glob.glob(os.path.abspath(args.audio_path) + "/*.wav", recursive=True)
mp3_paths = glob.glob(os.path.abspath(args.audio_path) + "/*.mp3", recursive=True)
ogg_paths = glob.glob(os.path.abspath(args.audio_path) + "/*.ogg", recursive=True)
dataPaths = []
dataPaths.extend(wav_paths)
dataPaths.extend(mp3_paths)
dataPaths.extend(ogg_paths)
print("load dataset paths done. {}files".format(dataPaths.__len__()))

for path in dataPaths:
    filename = os.path.basename(path)
    label ,fs= librosa.load(path, sr=util.sample_ratio)
    if  args.audio_dataset_second is not None:
        label = util.clip_audio_length(label,fs,second=args.audio_dataset_second)

    # preprocess audio
    input = util.generate_inputWave(fs, label)
    label_abs, label_phase = util.convert_to_spectrogram(label)
    input_abs,input_phase = util.convert_to_spectrogram(input)
    label_abs,_scale_factor,_offset= util.scaleArray(label_abs)
    input_abs,_scale_factor,_offset= util.scaleArray(input_abs)

    # merge&save AtoB image -------------
    #  create image array            -> (channel,row,col)
    labelImg = (np.array([label_abs,label_phase,np.zeros(label_abs.shape)])*255).astype('uint8')
    inputImg = (np.array([input_abs,input_phase,np.zeros(input_abs.shape)])*255).astype('uint8')
    # reorder to channel last       -> (row,col,channel)
    labelImg = np.rollaxis(labelImg,0,3)
    inputImg= np.rollaxis(inputImg,0,3)
    # concatenate horizontary
    image_array = np.hstack((labelImg,inputImg))
    scipy.misc.imsave(os.path.join(args.output_path,filename)+".png", image_array)

    print("processed {}".format(filename))

