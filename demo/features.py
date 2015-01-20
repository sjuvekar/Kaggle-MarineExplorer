import sys
import numpy as np
import fileio
from matplotlib import mlab, pyplot

if __name__ == "__main__":
    img = fileio.ReadAIFF("../data/train/train{}.aiff".format(sys.argv[1]))
    P, freqs, bins = mlab.specgram(img, NFFT=256, Fs=2000, noverlap=192)

    maxM = 60
    inner = 3
    outer = 10

    Q = P.copy()
    m, n = Q.shape
    mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
    fact_ = 1.5
    Q[Q > mval + fact_*sval] = mval + fact_*sval
    Q[Q < mval - fact_*sval] = mval - fact_*sval

    R = Q.copy()
    S = Q.copy()
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    for i in range(maxM):
        R[i,:] = np.convolve(R[i,:],wOuter,'same')
        S[i,:] = np.convolve(S[i,:],wInner,'same')
        Q[i,:] = Q[i,:] - (np.convolve(Q[i,:],wOuter,'same') - np.convolve(Q[i,:],wInner,'same'))/(outer-inner)

    pyplot.subplot(1, 4, 1)
    pyplot.imshow(P)
    pyplot.title("Original")

    pyplot.subplot(1, 4, 2)
    pyplot.imshow(R[:maxM,:])
    pyplot.title("Outer")

    pyplot.subplot(1, 4, 3)
    pyplot.imshow(S[:maxM,:])
    pyplot.title("Inner")

    pyplot.subplot(1, 4, 4)
    pyplot.imshow(Q[:maxM,:])
    pyplot.title("Convolve")

    pyplot.show()
    print Q[:maxM, :].shape

    inner=3
    outer=10
    maxM = 40
   
    Q = P.copy()
    m, n = Q.shape
    mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
    fact_ = 1.5
    Q[Q > mval + fact_*sval] = mval + fact_*sval
    Q[Q < mval - fact_*sval] = mval - fact_*sval

    R = Q.copy()
    S = Q.copy()
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    for i in range(n):
      R[:,i] = np.convolve(Q[:,i],wOuter,'same')
      S[:,i] = np.convolve(Q[:,i],wInner,'same')
      Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer-inner)

    pyplot.subplot(1, 4, 1)
    pyplot.imshow(P)
    pyplot.title("Original")

    pyplot.subplot(1, 4, 2)
    pyplot.imshow(R[:maxM,:])
    pyplot.title("Outer")

    pyplot.subplot(1, 4, 3)
    pyplot.imshow(S[:maxM,:])
    pyplot.title("Inner")

    pyplot.subplot(1, 4, 4)
    pyplot.imshow(Q[:maxM,:])
    pyplot.title("Convolve")

    pyplot.show()
    print Q[:maxM, :].shape
   

