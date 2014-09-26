
import pickle
import pandas as pd
from sys import argv

results = pickle.load(open(argv[1]))



# print "Keys:      ", results[0].keys()
# print "StatsKeys: ", results[0]['stats'].keys()

tbl = []

for r in results:
    #    tbl += [(r['pyConfig']['dataset'], r['algorithm'], r['runtimeMS'], \
    tbl += [(r['pyConfig']['algorithm'], \
             r['runtimeMS'], \
             r['pyConfig']['dataset'], \
             r['iterations'], \
             r['objective'], \
             r['loss'], \
             r['reg'], \
             r['propError'], \
             r['pyConfig']['ADMMlagrangianRho'], \
             r['pyConfig']['ADMMrho'], \
             r['stats']['consensusNorm'] if 'consensusNorm' in r['stats'] else -1, \
             r['stats']['primalAvgNorm'] if 'primalAvgNorm' in r['stats'] else -1, \
             r['stats']['dualAvgNorm'] if 'dualAvgNorm' in r['stats'] else -1, \
             r['stats']['avgSGDIters'] if 'avgSGDIters' in r['stats'] else 0, \
             r['stats']['avgMsgsSent'] if 'avgMsgsSent' in r['stats'] else 0, \
             r['stats']['avgMsgsRcvd'] if 'avgMsgsRcvd' in r['stats'] else 0)]
 #, r['stats']['avgSGDIters'])]
#    print r

#print tbl
columns = ['alg', 'Runtime', 'dataset', 'Iters', '*Obj*', 'Loss', 'Reg', 'Error', \
           'lrho', 'Rho', 'consensusNorm', 'primalNorm', 'dualNorm',  'SGDItres', 'msgsSent', 'msgsRcv']

frame = pd.DataFrame.from_records(tbl, columns= columns)
print frame.to_string()
