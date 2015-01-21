import pandas
import numpy

TEST_FRAC = 0.15
df = pandas.read_csv("total.csv")
df[df.columns[1:]] = (df[df.columns[1:]] + 255.) / 255.
l = len(df)
test_len = int(l * TEST_FRAC)

idx = list(df.index)
numpy.random.shuffle(idx)

test = df.ix[idx[0:test_len]]
test.to_csv("img_test.csv", index=False)

valid = df.ix[idx[test_len:2*test_len]]
valid.to_csv("img_valid.csv", index=False)

train = df.ix[idx[2*test_len:]]
train.to_csv("img_train.csv", index=False)

