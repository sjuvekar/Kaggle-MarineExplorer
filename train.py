from pylearn2.config import yaml_parse

if __name__ == "__main__":
   clf = yaml_parse.load_path("conv.yaml")
   clf.main_loop()  
