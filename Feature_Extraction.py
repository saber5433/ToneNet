import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile

def melspectrogram_feature(audio_path, save_path):
    sr, y = scipy.io.wavfile.read(audio_path)
    if y.dtype!='float32':
            y = y.astype('float32') / 32767.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,n_fft=2048, hop_length=16,
                                       fmin=50,fmax=350)
    
    plt.figure(figsize=(2.25, 2.25))
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                             sr=sr,
                              fmax=350)
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    name = os.path.basename(audio_path).split('.')[0] + '.jpg'
    plt.savefig(os.path.join(save_path, name))
    #plt.show()

    plt.close('all')
    return os.path.join(save_path, name)


