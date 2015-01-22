#!/usr/bin/env python
import numpy
import aifc
import pandas
import scipy 
from sklearn import preprocessing

f = pandas.read_csv("data/train.csv")
trainY = f["label"]

train_dim = 30000
test_dim = 54503
nframes = 4000

# Read Train Data
trainX = numpy.zeros(train_dim * nframes).reshape(train_dim, nframes)
for i in range(0, train_dim):
  filename = "data/train/train%d.aiff" % (i+1)
  print filename
  f = aifc.open(filename, "r")
  strsig = f.readframes(nframes)
  f.close()
  x = numpy.fromstring(strsig, numpy.short).byteswap()
  y = 1. / nframes * numpy.abs(scipy.fft(x))
  trainX[i, :] = y

# Read test data
testX = numpy.zeros(test_dim * nframes).reshape(test_dim, nframes)
for i in range(0, test_dim):
  filename = "data/test/test%d.aiff" % (i+1)
  print filename
  f = aifc.open(filename, "r")
  strsig = f.readframes(nframes)
  f.close()
  x = numpy.fromstring(strsig, numpy.short).byteswap()
  y = 1. / nframes * numpy.abs(scipy.fft(x))
  testX[i, :] = y

# Scale train and test
print "Computing scaler"
scaler = preprocessing.StandardScaler().fit(trainX)
print "Scaling trainX"
trainX = scaler.transform(trainX)
print "Scaling testX"
testX = scaler.transform(testX)


