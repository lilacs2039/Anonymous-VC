#!/usr/bin/env python
"audio signal processing."

import librosa
import numpy as np
import pyworld as pw
import scipy
import imageio
from sklearn.preprocessing import normalize


class conf:
    """
    Configuration Parameter Class
    """

    """
    time length of preprocessed audio  
    """
    prep_audio_dataset_second=3

    # sampling rate
    sample_ratio = 16000 # 22050
    """
    Short Time FFT window size

    librosa default value　2048
    stft returned value shape　(1 + n_fft/2, t)
    """
    n_fft = 256 #2048

    """
    Encoderで縦・横方向に畳み込むサイズ倍率
    Encが8レイヤ、各レイヤで行列サイズ1/2になるので 256

    入力スペクトログラムの行・列のサイズはこの倍数とすること
    """
    Encocer_Feature_Constant=2**7   #2**7:128@n_fft256  #2**8:256

    """
    enable saving the label(specImage)
    """
    enable_output_labelWav = True

    """
    scaleArray()のscaleFactorを表示するか
    """
    print_scaleFactor=False

    """
    スペクトログラムへの変換時の最小強度
    　→　対数とったときに-6が最小値になる
    """
    eps= 10**-6
    
    """
    強度スペクトログラムの正規化時のスケール倍率
    """
    scale_abs=0.1
    """
    強度スペクトログラムの正規化時のオフセット
    epsから決定
    """
    offset_abs=0.6

    """
    位相スペクトログラムの正規化時のスケール倍率
    """
    scale_phase=1/(np.pi*2)
    """
    位相スペクトログラムの正規化時のオフセット
    """
    offset_phase=0.5


def convert_to_wave(Dabs, Dphase):
    D_hat = 10 ** Dabs * np.exp(1j*Dphase)    #xp.exp(1j*Dphase)
    y_hat = librosa.istft(D_hat)
    return y_hat

def convert_to_spectrogram(waveNDArray):
    """
    convert audio 1D Numpy Array to spectrogram 2D Numpy Array.
    note. Dabs = np.log10(np.abs(D) + 10**-6)

    :param waveNDArray:
    :return: Dabs,Dphase
    """
    # スペクトル・位相マップ　作成
    D = librosa.stft(waveNDArray, n_fft=conf.n_fft)  #D:np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
    Dabs = np.log10(np.abs(D) + conf.eps)
    Dphase = np.angle(D)
    return Dabs,Dphase

def padding_spectrogram(D):
    """
    スペクトログラムの行列サイズをEncoderに特徴的な値の整数倍にする
    Encが8レイヤ、各レイヤで行列サイズ1/2になるので、入力スペクトログラムの行・列のサイズは256の倍数とする
    """
    D = D[0:D.shape[0]-1,:]  #最後の行を削除 TODO: 0:-1
    w_div,w_rem = divmod(D.shape[1], conf.Encocer_Feature_Constant)
    D = np.pad(D, [(0,0), (0, conf.Encocer_Feature_Constant * (w_div + 1) - D.shape[1])],
               'constant', constant_values = np.min(np.abs(D)))
    return D

def anonymization(fs, waveNDArray, f0Value = 0, sp_strechRatio = np.random.uniform(0.6, 2, size=1), gaussian_s = 3):
    """
    WAV音声データから話者情報を取り除いたWAV音声データを作成
    label音声からinput音声作成用
    :param path:
    :param f0Value:
    :param sp_strechRatio:
    :return:
    """
    waveNDArray = waveNDArray.astype(np.float)
    _f0, t = pw.dio(waveNDArray, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(waveNDArray, _f0, t, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(waveNDArray, f0, t, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(waveNDArray, f0, t, fs)  # 非周期性指標の抽出
    f0_fixed0 = np.ones(f0.shape) * f0Value
    f0_median = np.median(f0)
    sp_median = np.median(sp)
    ap_median = np.median(ap)
    # SPを高周波方向に伸縮
    sp2 = np.ones_like(sp)*np.min(sp)
    for f in range(sp2.shape[1]):
        if(int(f / sp_strechRatio) >= sp.shape[1]): break
        sp2[:, f] = sp[:, int(f / sp_strechRatio)]
    # SP/APに正規分布ノイズ
    sp_noised = sp2 + np.random.normal(sp_median,sp_median/10,sp2.shape)
    ap_noised = ap + np.random.normal(ap_median,ap_median/10,ap.shape)
    #ガウシアンフィルタ
    sp_gaussian = scipy.ndimage.filters.gaussian_filter(sp_noised,gaussian_s)
    ap_gaussian = scipy.ndimage.filters.gaussian_filter(ap_noised,gaussian_s)
    # 音声復元
    synthesized = pw.synthesize(f0_fixed0, sp, ap, fs)
    return synthesized


def normalize_abs(a:np.ndarray) ->(np.ndarray,float,float):
    return normalize(a,scale=conf.scale_abs,offset=conf.offset_abs)
def normalize_phase(a:np.ndarray)->(np.ndarray,float,float):
    return normalize(a,scale=conf.scale_phase,offset=conf.offset_phase)

def normalize(ndArray, min=0, max=1,scale=None,offset=None):
    """
    normalize ndArray.
    (all ndArray values are clamped in min~max.)
    :param ndArray:
    :param min:
    :param max:
    :return: スケール後の配列、スケール倍率、オフセット
    """
    if scale==None:
        scale = (max-min) / (np.max(ndArray) - np.min(ndArray))
    scaled = ndArray * scale
    if offset==None:
        offset = - np.min(scaled) + min
    ret = scaled + offset
    if(ret.min()<min) or (max<ret.max()):
        print("warning:normalized value outrange (but cliped).check scale/offset value.")
        print(f"original max/min:{ndArray.max()}/{ndArray.min()}")
        print(f"original max/min:{ndArray.max()}/{ndArray.min()}")
    if conf.print_scaleFactor:
        print('scale:{}, offset:{}'.format(scale,offset))
    return ret.clip(min,max), scale, offset

def denormalize_abs(a:np.ndarray)->np.ndarray:
    return denormalize(a,scale=conf.scale_abs,offset=conf.offset_abs)
def denormalize_phase(a:np.ndarray) ->np.ndarray:
    return denormalize(a,scale=conf.scale_phase,offset=conf.offset_phase)
def denormalize(ndArray:np.ndarray, scale:float,offset:float) ->np.ndarray:
    return (ndArray - offset)/ scale

def clip_audio_length(audio_ndarray, sr, second):
    """
    audio_ndarray の長さ[秒]をsecondになるようにカット・paddingする
    :param audio_ndarray:
    :param sr:
    :return:
    """
    if audio_ndarray.shape[0] > second * sr:
        ret = audio_ndarray[:second * sr]
    else:
        ret = np.pad(audio_ndarray, [(0, second * sr - audio_ndarray.shape[0])], 'constant', constant_values=0)
    assert ret.__len__() == second * sr , "audioのサイズが second[sec] * sr(sampling rate)[/sec]になっていない"
    return ret

def save_spectrogram_asImage(Dabs,Dphase,savename):
    """
    save ndarray matrix as image.
      r:abs
      g:phase
      b:none
    abs/phase must be clamped in 0~1.
    :param Dabs:
    :param Dphase:
    :param savename: includes image extension (ex. '.png')
    :return:
    """
    #  create image array            -> (channel,row,col)
    assert (0 <= Dabs.min()) & ( Dabs.max() <=1), f"Dabs must be in 0~1. min:{Dabs.min()}, max:{Dabs.max()}"
    assert (0 <= Dphase.min()) & ( Dphase.max() <=1) , "Dphase must be in 0~1"
    srcImg = (np.array([Dabs,Dphase,np.zeros(Dabs.shape)])*255).astype('uint8')
    # reorder to channel last       -> (row,col,channel)
    srcImg = np.rollaxis(srcImg,0,3)
    imageio.imsave(savename, srcImg)

