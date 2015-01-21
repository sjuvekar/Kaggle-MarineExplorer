import pandas
import numpy

TEST_FRAC = 0.15
df = pandas.read_csv("total.csv")
l = len(df)
test_len = int(l * test_frac)

idx = list(df.index)
numpy.random.shuffle(idx)

test = df.ix[idx[0:test_len]]
test.to_csv("img_test.csv", index=False)

valid = df.ix[idx[test_len:2*test_len]]
valid.to_csv("img_valid.csv", index=False)

train = df.ix[idx[2*test_len:]]
train.to_csv("img_train.csv", index=False)

