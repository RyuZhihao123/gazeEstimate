import pickle
from Network.Config import Configure
dasidas = pickle.load(open(Configure.PATH_CUSTOMIZED_GT,'rb'))

print(dasidas)