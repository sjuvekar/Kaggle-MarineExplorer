import metrics
import fileio
import numpy
from matplotlib import mlab,pyplot as plt
import pandas

df = pandas.DataFrame(numpy.zeros(30000*5901).reshape(30000, 5901), columns=(["label"]+map(lambda a:str(a), range(100 * 59))))

labels =pandas.read_csv("../data/train.csv")

for i in range(30000):
  img = fileio.ReadAIFF("../data/train/train{}.aiff".format(i+1))
  P, freqs, bins = mlab.specgram(img, NFFT=256, Fs=2000, noverlap=192)
  Q = metrics.slidingWindowV(P,inner=3,maxM=40)
  W = metrics.slidingWindowH(P,inner=3,outer=32,maxM=60)
  s = numpy.vstack( (Q, W) )
  df.ix[i] = numpy.append(labels["label"][i], s.reshape(100 * 59))
  if i % 100 == 0:
    print i
df.to_csv("total.csv", index=False)
