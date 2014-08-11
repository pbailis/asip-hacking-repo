
import pickle

for r in pickle.load(open("experiment.pkl")):
    print r['pyConfig']['dataset'], r['algorithm'][:15].ljust(15),  r['runtimeMS'], ": \t", r['iterations'], "\t", r['totalLoss'], "\t", r['trainingError'], "\t", r['roc']
