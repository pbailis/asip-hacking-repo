
import pickle
from pylab import *
from sys import argv
import glob

logLoss = False
yval = 'objective'
matplotlib.rcParams['figure.figsize'] = 10, 7#3.5, 1.7

fnRegCombos = [("SVM", "L2"), ("SVM", "L1"), ("LR", "L2"), ("LR", "L1")]

if len(argv) < 2:
    pickle_filename = "08_08_14_overnight.pkl"
else:
    pickle_filename = argv[1]

results = []

for r in glob.glob(pickle_filename):
    results += pickle.load(open(r))
    print r

#results = [r for r in results if r['algorithm'] != "HOGWILD"]

print results[0].keys()
print results[0]['pyConfig'].keys()

# detect legacy data and update fields
if "runtime_ms" not in results[0].keys():
    print "New pyconfig keys", results[0]['pyConfig'].keys()
    for r in results:
        r['algorithm'] = r['pyConfig']['algorithm']
        r['runtime_ms'] = r['runtimeMS']
        r['command'] = r['pyConfig']['command']
        r['dataset'] = r['pyConfig']['dataset']
        r['pointCloudDim'] = r['pyConfig']['pointCloudDim']
        r['pointCloudSkew'] = r['pyConfig']['pointCloudSkew']
        r['total_loss'] = r['loss']
        r['objective'] = r['objective']
        r['objFn'] = r['pyConfig']['objective']
        r['regType'] = r['pyConfig']['regType']
    


datasets = unique([r['dataset'] for r in results if r['dataset'] != 'cloud'])

print datasets

for (objFn, regType) in fnRegCombos:
    print objFn, regType
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset and r['objFn'] == objFn and r['regType'] == regType]
        algs = unique([r['algorithm'] for r in dataset_results])
        print "Dataset: %s %s %s" % (dataset, objFn, regType)
        for alg in algs:
            alg_results = [r for r in dataset_results if r['algorithm'] == alg]
            plot_p = [(r['runtime_ms'], r[yval]) for r in alg_results]
            plot_p.sort(key = lambda x: x[0])
            plotx = [r[0] for r in plot_p]
            ploty = [r[1] for r in plot_p]

            for p in plot_p:
                print "\t", alg, p[0], p[1]
    
            plot(plotx, ploty, 'o-', label=alg)

        if logLoss:
            gca().set_yscale('log')
            #gca().set_xscale('log')

        if dataset == "wikipedia":
            ylim(ymax=390000, ymin=175000)

        legend()
        xlabel("Time (ms)")
        ylabel("Objective")
        title("%s-%s-%s" % (dataset, objFn, regType))
        savefig("%s-%s-%s.png" % (dataset, objFn, regType))
        cla()
        clf()

# POINT CLOUD PLOTS

datasets = unique([r['dataset'] for r in results if r['dataset'] == 'cloud'])

skews = unique([r['pointCloudSkew'] for r in results if r['dataset'] == 'cloud'])

for dataset in datasets:
    for skew in skews:
        dataset_results = [r for r in results if r['dataset'] == dataset and r['pointCloudSkew'] == skew]
        algs = unique([r['algorithm'] for r in dataset_results])
        print "Dataset: " , dataset, "skew", skew
        for alg in algs:
            alg_results = [r for r in dataset_results if r['algorithm'] == alg]
            plot_p = [(r['runtime_ms'], r[yval]) for r in alg_results]
            plot_p.sort(key = lambda x: x[0])
            plotx = [r[0] for r in plot_p]
            ploty = [r[1] for r in plot_p]

            for p in plot_p:
                print "\t", alg, p[0], p[1]
    
            plot(plotx, ploty, 'o-', label=alg)

        if logLoss:
            gca().set_yscale('log')
            #gca().set_xscale('log')

        legend()
        savefig(dataset + "-"+str(skew) +".pdf")
        cla()
        clf()
