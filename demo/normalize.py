import pandas

def normalize(filename, outputname):
	df = pandas.read_csv(filename, header=None)
	mn = df.min(axis=0)
	rng = df.max(axis=0) - df.min(axis=0)
	ss = (df - mn) / rng
	ss[0] = df[0]
	header = ["Label"] + map(lambda a:str(a), range(129 * 30))
	ss.columns = header
	ss.to_csv(outputname, index=False)

if __name__ == "__main__":
	for f in ["train.csv", "valid.csv", "test.csv"]:
		print f
		normalize(f, "img_"+f)	
