
import pickle
import pandas as pd
results = pickle.load(open("experiment.pkl"))

print "Keys:      ", results[0].keys()
print "StatsKeys: ", results[0]['stats'].keys()

tbl = []

for r in results:
    tbl += [(r['pyConfig']['dataset'], r['algorithm'], r['runtimeMS'], r['iterations'], r['totalLoss'],  r['trainingError'], r['roc'])] #, r['stats']['avgSGDIters'])]
#    print r

#print tbl

frame = pd.DataFrame.from_records(tbl, columns=['Data', 'Alg', 'Runtime', 'Iterations', 'Loss', 'Error', 'Roc']) #, 'SGDItres'])
print frame.to_string()
