import pickle
import sys
a= pickle.load(open(sys.argv[1],"rb"))
b= pickle.load(open(sys.argv[2],"rb"))
print((a==b).all())
