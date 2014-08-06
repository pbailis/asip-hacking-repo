
import pickle
from pylab import *
from sys import argv

logLoss = True

matplotlib.rcParams['figure.figsize'] = 6, 3#3.5, 1.7

if len(argv) < 2:
    pickle_filename = "08_06_14_newrun.pkl"
else:
    pickle_filename = argv[1]

results = pickle.load(open(pickle_filename))

results = [r for r in results if r['algorithm'] != "HOGWILDSVM"]

bismarck_results = [r for r in results if r['command'].find("bismarck") != -1]

algs = unique([r['algorithm'] for r in results])

print "BISMARCK"

for alg in algs:
    alg_results = [r for r in bismarck_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]

    for p in plot_p:
        print alg, p[0], p[1]
    
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')
#gca().set_xscale('log')

legend()
savefig("bismarck.pdf")

cla()

print "FLIGHTS"

flights_results = [r for r in results if r['dataset'] == "flights"]

for alg in algs:
    alg_results = [r for r in flights_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]

    for p in plot_p:
        print alg, p[0], p[1]
    
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')
#gca().set_xscale('log')

legend()
savefig("flights.pdf")

clf()

print "POINTCLOUD DIM 2"

cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 2]

algs = unique([r['algorithm'] for r in results])

for alg in algs:
    alg_results = [r for r in cloud_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]

    for p in plot_p:
        print alg, p[0], p[1]
    
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')
#gca().set_xscale('log')
legend()
savefig("pc-dim2.pdf")


clf()

print "POINTCLOUD DIM 100"

cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 100]

algs = unique([r['algorithm'] for r in results])

for alg in algs:
    alg_results = [r for r in cloud_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss'], r['command']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])

    for p in plot_p:
        print alg, p[0], p[1]#, p[2]
    
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')
#gca().set_yscale('log')
#gca().set_xscale('log')
legend()
savefig("pc-dim100.pdf")

clf()

print "POINTCLOUD DIM 10 SKEW 0"


cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 10 and r['pointCloudSkew'] == 0]

algs = unique([r['algorithm'] for r in results])

for alg in algs:
    alg_results = [r for r in cloud_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss'], r['command']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])

    for p in plot_p:
        print alg, p[0], p[1]
    
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')
#gca().set_yscale('log')
#gca().set_xscale('log')
legend()
savefig("pc-dim10-skew0.pdf")

cla()

print "POINTCLOUD DIM 10 SKEW 0.1"

cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 10 and r['pointCloudSkew'] == 0.1]


algs = unique([r['algorithm'] for r in results])

for alg in algs:
    alg_results = [r for r in cloud_results if r['algorithm'] == alg]

    plot_p = [(r['runtime_ms'], r['training_loss']) for r in alg_results]
    plot_p.sort(key = lambda x: x[0])

    for p in plot_p:
        print alg, p[0], p[1]
    
    plotx = [r[0] for r in plot_p]
    ploty = [r[1] for r in plot_p]
    plot(plotx, ploty, 'o-', label=alg)

if logLoss:
    gca().set_yscale('log')

    
#gca().set_yscale('log')
#gca().set_xscale('log')
legend()
savefig("pc-dim10-skewpoint1.pdf")
