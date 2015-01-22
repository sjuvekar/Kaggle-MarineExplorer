import cv2
import numpy
import fileio
import pandas
import random
import sys
from matplotlib import mlab
from matplotlib import pyplot as plt

import metrics

NUM_IMAGES=4
DATA_PATH="../Kaggle-MarineExplorer/data/"

def convert_spec(img):
    freqs = fileio.ReadAIFF(img)
    P, freqs, bins = mlab.specgram(freqs, NFFT=256, Fs=2000, noverlap=192)
    return P

def slidingWindowH(img):
    w = metrics.slidingWindowH(img, inner=3, outer=32, maxM=60)
    print w.shape
    return w

def slidingWindowV(img):
    w = metrics.slidingWindowV(img, inner=3, maxM=40)
    print w.shape
    return w

def highFrequencyConv(img, tmpl):
    Q = metrics.slidingWindowH(img,inner=7,maxM=50,norm=True)[20:32, :]
    mf = cv2.matchTemplate(Q.astype('Float32'), tmpl, cv2.TM_CCOEFF_NORMED)
    print mf.shape
    return mf

def add_img_to_plot(img_specgram, index):
    plt.subplot(1, NUM_IMAGES, index)
    plt.imshow(img_specgram)

def add_line_to_plot(img, index):
    plt.subplot(1, NUM_IMAGES, index)
    plt.hist(img.T)

if __name__ == "__main__":
    
    df = pandas.read_csv(DATA_PATH+"train.csv")
    value = int(sys.argv[1])
    examples = df[df["label"] == value]
    random_examples = random.sample(examples.index, NUM_IMAGES)
    print random_examples

    random_examples = [25216, 9480, 3465, 19321]

    # Define vertical highFreq bars
    bar_ = numpy.zeros((12,9),dtype='Float32')
    bar1_ = numpy.zeros((12,12),dtype='Float32')
    bar2_ = numpy.zeros((12,6),dtype='Float32')
    bar_[:,3:6] = 1.
    bar1_[:,4:8] = 1.
    bar2_[:,2:4] = 1.

    orig_plots = list()
    slidingH = list()
    slidingV = list()    
    templLeft = list()
    templMid = list()
    templRight = list()

    for i in range(len(random_examples)):
        idx = random_examples[i]
        clip = examples["clip_name"][idx]
        img = convert_spec(DATA_PATH+"train/"+clip)
        orig_plots.append(img)
        h_slided = slidingWindowH(img)
        slidingH.append(h_slided)
        v_slided = slidingWindowV(img)
        slidingV.append(v_slided)
        l_templ = highFrequencyConv(img, bar2_)
        templLeft.append(l_templ)
        r_templ = highFrequencyConv(img, bar1_)
        templRight.append(r_templ)
        m_templ = highFrequencyConv(img, bar_)
        templMid.append(m_templ)

    # plot orig
    for i in range(len(random_examples)):
        add_img_to_plot(orig_plots[i], i+1)
    plt.show()

    # plot slide
    for i in range(len(random_examples)):
        add_img_to_plot(slidingH[i], i+1)
    plt.show()

    # plot slide V
    for i in range(len(random_examples)):
        add_img_to_plot(slidingV[i], i+1)
    plt.show()

    # plot left Templ
    for i in range(len(random_examples)):
        add_line_to_plot(templLeft[i], i+1)
    plt.show()

    # plot rightTempl
    for i in range(len(random_examples)):
        add_line_to_plot(templRight[i], i+1)
    plt.show()

    # plot middleTempl
    for i in range(len(random_examples)):
        add_line_to_plot(templMid[i], i+1)
    plt.show()
