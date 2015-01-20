#!/usr/bin/env python
import gc
import numpy
import aifc
import glob
import pandas
import pylab
import scipy
import pickle
import math
import sys
from scipy import signal
from sklearn import preprocessing, cross_validation, ensemble


def remove_dup():
  seen = set()
  seen_names = []
  dupes = []
  for fn_id in range(1, 30001):
    fn = "../data/train/train%d.aiff" % fn_id
    contents = open(fn).read()
    if contents in seen:
      dupes.append(fn)
    else:
      seen.add(contents)
      seen_names.append(fn_id)
  return seen_names


def specgram(filename):
  # Define contants
  nframes = 4000
  print filename
  f = aifc.open(filename, "r")
  strsig = f.readframes(nframes)
  f.close()
  x = numpy.fromstring(strsig, numpy.short).byteswap()
  a = pylab.specgram(x)
  return a[0]

def specgram_large():
  nfft=256;
  nt=20;
  fs=2000;
  dt=1./fs;
  fmax=400;
  nframes = 4000

  # Read Data
  print filename
  f = aifc.open(filename, "r")
  strsig = f.readframes(nframes)
  f.close()
  x = numpy.fromstring(strsig, numpy.short).byteswap()
  n = len(x)
  
  # Initialize output
  nf=nfft/2.+1
  tt = numpy.arange(dt*(nfft-1)/2, (n-1)*dt-(nfft/2)*dt, nt * dt)
  ntt=len(tt)
  y = numpy.zeros(nf * ntt).reshape(nf, ntt);
 
  # Create Window vector
  xw = range(0, nfft)
  wind = map(lambda a: 0.5 * (1 - math.cos((a * 2 * math.pi) / (nfft - 1.0))), xw)
  for i in range(0, ntt):
    zi = numpy.arange(i*nt, nfft*(i+1) - i*(nfft-nt))
    xss = scipy.fft(x[zi] * wind, nfft)/nfft 
    yy = 2 * abs(xss[0:(nfft/2)+1])
    y[:, i] = yy

  # Return y
  return y.reshape(nf * ntt)


def filter_specgrams_train(seen):
  train_dim = len(seen)
  nfeatures = 129 * 30

  # First create trainY
  f = pandas.read_csv("../data/train.csv")
  trainY = f["label"]
  trainY = trainY.ix[map(lambda x: x - 1, seen)]

  # Next read Train data
  trainX = numpy.zeros(train_dim * nfeatures).reshape(train_dim, nfeatures)
  for i in range(0, len(seen)):
    sp = specgram("../data/train/train%d.aiff" % seen[i]) 
    y = signal.wiener(sp)
    print y.shape
    trainX[i, :] = y.reshape(nfeatures)

  return (trainX, trainY)


def filter_specgrams_test():
  test_dim = 54503
  nfeatures = 129 * 30 

  testX = numpy.zeros(test_dim * nfeatures).reshape(test_dim, nfeatures)
  for i in range(0, test_dim): 
    sp = specgram("../data/test/test%d.aiff" % (i+1))
    y = signal.wiener(sp)
    print y.shape
    testX[i, :] = y.reshape(nfeatures)
  return testX


if __name__ == "__main__":
  print "Removing Duplicates"
  seen = remove_dup()
  print len(seen)

  # Read train data and fit model
  (trainX, trainY) = filter_specgrams_train(seen)
  print "trainX: " + str(trainX.shape)
  print "trainY: " + str(trainY.shape) 
  # Read test data and fit model
  testX = filter_specgrams_test()
  print "testX: " + str(testX.shape)
 
  # Scale train and test
  print "Computing scaler"
  scaler = preprocessing.StandardScaler().fit(trainX)
  print "Scaling trainX"
  trainX = scaler.transform(trainX)
  print "Scaling testX"
  testX = scaler.transform(testX)

  t = ensemble.GradientBoostingClassifier(n_estimators=int(sys.argv[1]), verbose=1, max_depth=6, min_samples_leaf=5)
  t.fit(trainX, trainY)
  p = t.predict_proba(testX)
  numpy.savetxt(sys.argv[2], p[:, 1].astype(float), fmt='%1.10f', delimiter=",")

