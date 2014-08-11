
import pickle

for r in pickle.load(open("experiment.pkl")):
    print r['pyConfig']['dataset'], r['algorithm'], r['runtimeMS'], r['iterations'], r['totalLoss'], r['pr'], r['roc']
