
import pickle
import pandas as pd
from sys import argv

results = pickle.load(open(argv[1]))

#print "Keys:      ", results[0].keys()
#print "StatsKeys: ", results[0]['stats'].keys()

tbl = []

for r in results:
    #    tbl += [(r['pyConfig']['dataset'], r['algorithm'], r['runtimeMS'], \
    tbl += [(r['algorithm'], \
             r['runtimeMS'], \
             r['pyConfig']['dataset'], \
             r['iterations'], r['totalLoss'],  r['trainingError'], r['roc'], \
             r['pyConfig']['ADMMlagrangianRho'], \
             r['pyConfig']['ADMMrho'], \
             r['regPenalty'], r['trainingLoss'], \
             r['stats']['avgSGDIters'] if 'avgSGDIters' in r['stats'] else 0, \
             r['stats']['avgMsgsRcvd'] if 'avgMsgsRcvd' in r['stats'] else 0)]
 #, r['stats']['avgSGDIters'])]
#    print r

#print tbl
columns = ['alg', 'Runtime', 'dataset', 'Iters', 'Loss', 'Error', \
           'Roc', 'lrho', 'Rho', 'reg', 'trainL', 'SGDItres', 'msgsRcv']

frame = pd.DataFrame.from_records(tbl, columns= columns)
print frame.to_string()
