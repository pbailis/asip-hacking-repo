
import pickle
from pylab import *
from sys import argv

logLoss = False
yval = 'totalLoss'
matplotlib.rcParams['figure.figsize'] = 10, 7#3.5, 1.7

if len(argv) < 2:
    pickle_filename = "08_08_14_overnight.pkl"
else:
    pickle_filename = argv[1]

results = pickle.load(open(pickle_filename))

#results = [r for r in results if r['algorithm'] != "HOGWILD"]

print results[0].keys()

# detect legacy data and update fields
if "runtime_ms" not in results[0].keys():
    print "New pyconfig keys", results[0]['pyConfig'].keys()
    for r in results:
        r['runtime_ms'] = r['runtimeMS']
        r['command'] = r['pyConfig']['command']
        r['dataset'] = r['pyConfig']['dataset']
        r['pointCloudDim'] = r['pyConfig']['pointCloudDim']
        r['pointCloudSkew'] = r['pyConfig']['pointCloudSkew']
        r['total_loss'] = r['totalLoss']



datasets = unique([r['dataset'] for r in results if r['dataset'] != 'cloud'])

print datasets

for dataset in datasets:
    dataset_results = [r for r in results if r['dataset'] == dataset]
    algs = unique([r['algorithm'] for r in dataset_results])
    print "Dataset: " , dataset
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
    savefig(dataset + ".pdf")
    cla()
    clf()


pcDimAndSkew = [(r['pointCloudDim'], r['pointCloudSkew']) for r in results if r['dataset'] == 'cloud']
seen = set()
pcDimAndSkew = [item for item in pcDimAndSkew if item not in seen and not seen.add(item)]

print pcDimAndSkew

for (dim, skew) in pcDimAndSkew:
    print "POINTCLOUD DIM", dim, " AND SKEW", skew

    cloud_results = [r for r in results if r['dataset'] == "cloud" \
                     and r['pointCloudDim'] == dim and r['pointCloudSkew'] == skew]

    algs = unique([r['algorithm'] for r in results])

    for alg in algs:
        alg_results = [r for r in cloud_results if r['algorithm'] == alg]
        
        plot_p = [(r['runtime_ms'], r[yval]) for r in alg_results]
        plot_p.sort(key = lambda x: x[0])
        plotx = [r[0] for r in plot_p]
        ploty = [r[1] for r in plot_p]

        for p in plot_p:
            print "\t", alg, p[0], p[1]
    
        plot(plotx, ploty, 'o-', label=alg)

    if logLoss:
        gca().set_yscale('log')
    
    legend()
    savefig("pc-dim_" + str(dim) +"_skew_" + str(skew) +".pdf")
    cla()
    clf()

# print "POINTCLOUD DIM 10 SKEW 0"


# cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 10 and r['pointCloudSkew'] == 0]

# algs = unique([r['algorithm'] for r in results])

# for alg in algs:
#     alg_results = [r for r in cloud_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r[yval], r['command']) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])

#     for p in plot_p:
#         print alg, p[0], p[1]
    
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]
#     plot(plotx, ploty, 'o-', label=alg)

# if logLoss:
#     gca().set_yscale('log')
# #gca().set_yscale('log')
# #gca().set_xscale('log')
# legend()
# savefig("pc-dim10-skew0.pdf")

# cla()

# print "POINTCLOUD DIM 10 SKEW 0.1"

# cloud_results = [r for r in results if r['dataset'] == "cloud" and r['pointCloudDim'] == 10 and r['pointCloudSkew'] == 0.1]


# algs = unique([r['algorithm'] for r in results])

# for alg in algs:
#     alg_results = [r for r in cloud_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r[yval]) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])

#     for p in plot_p:
#         print alg, p[0], p[1]
    
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]
#     plot(plotx, ploty, 'o-', label=alg)

# if logLoss:
#     gca().set_yscale('log')

    
# #gca().set_yscale('log')
# #gca().set_xscale('log')
# legend()
# savefig("pc-dim10-skewpoint1.pdf")



# bismarck_results = [r for r in results if r['command'].find("bismarck") != -1]

# algs = unique([r['algorithm'] for r in results])

# print "BISMARCK"

# for alg in algs:
#     alg_results = [r for r in bismarck_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r[yval]) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]

#     for p in plot_p:
#         print alg, p[0], p[1]
    
#     plot(plotx, ploty, 'o-', label=alg)

# #if logLoss:
# #    gca().set_yscale('log')
# #gca().set_xscale('log')

# legend()
# savefig("bismarck.pdf")

# cla()


# print "Wikipedia"

# wiki_results = [r for r in results if r['dataset'] == "wikipedia"]

# algs = unique([r['algorithm'] for r in results])

# for alg in algs:
#     alg_results = [r for r in wiki_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r['training_error']) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])

#     for p in plot_p:
#         print alg, p[0], p[1]
    
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]
#     plot(plotx, ploty, 'o-', label=alg)

# if logLoss:
#     gca().set_yscale('log')

    
# #gca().set_yscale('log')
# #gca().set_xscale('log')
# legend()
# savefig("wikipedia.pdf")
 

# cla()



# print "FLIGHTS"

# flights_results = [r for r in results if r['dataset'] == "flights"]

# for alg in algs:
#     alg_results = [r for r in flights_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r[yval],  r['training_error']) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]

#     for p in plot_p:
#         print alg, p[0], p[1]
    
#     plot(plotx, ploty, 'o-', label=alg)

# if logLoss:
#     gca().set_yscale('log')
# #gca().set_xscale('log')

# legend()
# savefig("flights.pdf")

# cla()

# print "DBLP"

# flights_results = [r for r in results if r['dataset'] == "dblp"]

# for alg in algs:
#     alg_results = [r for r in flights_results if r['algorithm'] == alg]

#     plot_p = [(r['runtime_ms'], r[yval], r['training_error']) for r in alg_results]
#     plot_p.sort(key = lambda x: x[0])
#     plotx = [r[0] for r in plot_p]
#     ploty = [r[1] for r in plot_p]

#     for p in plot_p:
#         print alg, p[0], p[1], p[2]
    
#     plot(plotx, ploty, 'o-', label=alg)

# #if logLoss:
# #    gca().set_yscale('log')
# #gca().set_xscale('log')

# legend()
# savefig("dblp.pdf")

