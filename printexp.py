
import pickle

for r in pickle.load(open("experiment.pkl")):
    print r['dataset'], r['algorithm'], r['runtime_ms'], r['total_loss']
