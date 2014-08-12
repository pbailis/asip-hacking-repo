
import pickle
import pandas as pd
results = pickle.load(open("experiment.pkl"))

print "Keys:      ", results[0].keys()
print "StatsKeys: ", results[0]['stats'].keys()

tbl = []

for r in results:
    #    tbl += [(r['pyConfig']['dataset'], r['algorithm'], r['runtimeMS'], \
    tbl += [(r['algorithm'], \
             r['runtimeMS'], \
             r['iterations'], r['totalLoss'],  r['trainingError'], r['roc'], \
             r['pyConfig']['ADMMlagrangianRho'], \
             r['pyConfig']['ADMMrho'], \
             r['regPenalty'], r['trainingLoss'], \
             r['stats']['avgSGDIters'] if 'avgSGDIters' in r['stats'] else 0, \
             r['stats']['avgMsgsRcvd'], \
             r['stats']['avgMsgsSent'], \
             r['stats']['primalAvgNorm'] if 'avgSGDIters' in r['stats'] else 0, \
             r['stats']['dualAvgNorm'] if 'avgSGDIters' in r['stats'] else 0, \
             r['stats']['consensusNorm'] if 'avgSGDIters' in r['stats'] else 0) ]

 #, r['stats']['avgSGDIters'])]
#    print r

#print tbl
columns = ['alg', 'Runtime', 'Iters', 'Loss', 'Error', \
           'Roc', 'lrho', 'Rho', 'reg', 'trainL', 'SGDItres', 'msgsRcv', 'msgsSent', \
           'primalAvgNorm', 'dualAvgNorm', 'wNorm']

frame = pd.DataFrame.from_records(tbl, columns= columns)
print frame.to_string()
