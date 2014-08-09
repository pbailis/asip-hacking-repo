
from os import system
from sys import exit
import pickle

## START OF EXPERIMENTAL PARAMETERS

RUNTIMES = [1000, 5000, 10000, 20000, 40000, 80000, 120000]

ALGORITHMS = ["ADMM", "MiniBatchADMM", "AsyncADMM", "HOGWILD", "GD", "PORKCHOP"]

PICKLED_OUTPUT = "experiment.pkl"

## END OF EXPERIMENTAL PARAMETERS


## START OF CONSTANTS

GLOBAL_ADMMepsilon = 0.00000001
GLOBAL_ADMMlocalEpsilon = 0.000001
GLOBAL_ADMMrho = 1.0
GLOBAL_ADMMlagrangianRho = 1.0

GLOBAL_ADMM_maxLocalIterations = 100000

GLOBAL_MiniBatchADMM_maxLocalIterations = 100
GLOBAL_MiniBatchADMM_localEpsilon = 0.000001

GLOBAL_HOGWILD_maxLocalIterations = 10
GLOBAL_HOGWILD_broadcastDelay = 10

GLOBAL_AsyncADMM_maxLocalIterations = 10000
GLOBAL_AsyncADMM_broadcastDelay = 100

GLOBAL_inputTokenHashKernelDimension = 100

## END OF CONSTANTS


## START OF DATASET FORMATTING

def describe_point_cloud(pointsPerPartition = 500000,
                         partitionSkew = 0.00,
                         labelNoise = 0.05,
                         dimension = 100):
    return   "--pointCloudPointsPerPartition %d " \
             "--pointCloudPartitionSkew %f " \
             "--pointCloudLabelNoise %f " \
             "--pointCloudDimension %d " % \
             (pointsPerPartition,
              partitionSkew,
              labelNoise,
              dimension)

def describe_forest():
    return " --input /user/root/bismarck_data/forest* "

def describe_flights(year):
    return " --input /user/root/flights/"+str(year)+".csv"

def describe_dblp():
    return " --input /user/root/dblp/binarized-year-to-title.txt"

def describe_wikipedia():
    return " --input /user/root/wiki/en-wiki-8-7-2014-tokenized.txt"

## END OF DATASET FORMATTING


## START OF TEST RUNNING CODE

def runTest(runtimeMS,
            algorithm,
            datasetName,
            ADMMepsilon = GLOBAL_ADMMepsilon,
            ADMMlocalEpsilon = GLOBAL_ADMMlocalEpsilon,
            ADMMmaxLocalIterations = 1000,
            ADMMrho = GLOBAL_ADMMrho,
            ADMMlagrangianRho = GLOBAL_ADMMlagrangianRho,
            regType="L2",
            regParam=0.0001,
            numPartitions = (8*16),
            broadcastDelay = 100,
            cloudDim=-1,
            cloudPartitionSkew=-1,
            flightsYear = "2008",
            wikipediaTargetWordToken = 4690,
            dblpSplitYear = 2007,
            inputTokenHashKernelDimension = 100,
            miscStr = ""):
    if datasetName == "bismarck":
        datasetConfigStr = describe_forest()
    elif datasetName == "cloud":
        datasetConfigStr = describe_point_cloud(partitionSkew = cloudPartitionSkew, dimension = cloudDim)
    elif datasetName == "flights":
        datasetConfigStr = describe_flights(flightsYear)
    elif datasetName == "dblp":
        datasetConfigStr = describe_dblp()
    elif datasetName == "wikipedia":
        datasetConfigStr = describe_wikipedia()
    else:
        print "Unknown dataset!"
        raise

    cmd = "cd /mnt/spark; sbin/stop-all.sh; sleep 5; sbin/start-all.sh; sleep 3;" \
          "./bin/spark-submit " \
          "--driver-memory 52g " \
          "--class org.apache.spark.examples.mllib.research.SynchronousADMMTests " \
          "examples/target/scala-*/spark-examples-*.jar " \
          "--algorithm %s " \
          "--regType %s " \
          "--regParam %f " \
          "--format %s " \
          "--numPartitions %d " \
          "--runtimeMS %d " \
          "--ADMMmaxLocalIterations %d " \
          "--ADMMepsilon %f  " \
          "--ADMMLocalepsilon %f " \
          "--ADMMrho %f " \
          "--ADMMLagrangianrho %f " \
          "--broadcastDelayMs %d " \
          "--dblpSplitYear %d " \
          "--wikipediaTargetWordToken %d " \
          "--inputTokenHashKernelDimension %d " \
          " %s %s " % \
          (algorithm,
           regType,
           regParam,
           datasetName,
           numPartitions,
           runtimeMS,
           ADMMmaxLocalIterations,
           ADMMepsilon,
           ADMMlocalEpsilon,
           ADMMrho,
           ADMMlagrangianRho,
           broadcastDelay,
           dblpSplitYear,
           wikipediaTargetWordToken,
           inputTokenHashKernelDimension,
           datasetConfigStr,
           miscStr)

    print cmd

    system("eval '%s' > /tmp/run.out 2>&1" % (cmd))

    results = []
    # s"RESULT: ${params.algorithm} \t ${i} \t ${totalTimeMs} \t ${metrics.areaUnderPR()} \t ${metrics.areaUnderROC()} " +
    # s"\t ${trainingLoss} \t  ${regularizationPenalty} \t ${trainingLoss + regularizationPenalty} \t ${model.weights}"
    for line in open("/tmp/run.out"):
        if line.find("RESULT") != -1:
            line = line.split()
            record = {
                "algorithm": algorithm,
                "iterations": float(line[2]),
                "expected_runtime": float(line[3]),
                "runtime_ms": float(line[4]),
                "training_error": float(line[5]),
                "training_loss": float(line[6]),
                "reg_penalty": float(line[7]),
                "total_loss": float(line[8]),
                "model": line[9],
                "dataset": datasetName,
                "datasetConfigStr": datasetConfigStr,
                "line": line,
                "ADMMepsilon": ADMMepsilon,
                "ADMMlocalEpsilon": ADMMlocalEpsilon,
                "ADMMmaxLocalIterations": ADMMmaxLocalIterations,
                "ADMMrho": ADMMrho,
                "ADMMlagrangianRho": ADMMlagrangianRho,
                "broadcastDelay": broadcastDelay,
                "command": cmd,
                "regParam": regParam,
                "numPartitions": numPartitions,
                "regType": regType,
                "pointCloudDim": cloudDim,
                "pointCloudSkew": cloudPartitionSkew,
                "inputTokenHashKernelDimension": inputTokenHashKernelDimension,
                "dblpSplitYear": dblpSplitYear,
                "wikipediaTargetWordToken": wikipediaTargetWordToken
            }
            results.append(record)

    return results

results = []

## END OF TEST RUNNING CODE


## START OF EXPERIMENT RUNS

for dataset in ["wikipedia", "flights", "bismarck", "dblp"]:
    for runtime in RUNTIMES:
        for algorithm in ALGORITHMS:
            broadcastDelay = -1
            localEpsilon = GLOBAL_ADMMlocalEpsilon
            if algorithm == "ADMM":
                maxLocalIterations = GLOBAL_ADMM_maxLocalIterations
            elif algorithm == "MiniBatchADMM":
                maxLocalIterations = GLOBAL_MiniBatchADMM_maxLocalIterations
                localEpsilon = GLOBAL_MiniBatchADMM_localEpsilon
            elif algorithm == "HOGWILD":
                maxLocalIterations = GLOBAL_HOGWILD_maxLocalIterations
                broadcastDelay = GLOBAL_HOGWILD_broadcastDelay
            else:
                maxLocalIterations = GLOBAL_AsyncADMM_maxLocalIterations
                broadcastDelay = GLOBAL_AsyncADMM_broadcastDelay

            results += runTest(runtime,
                            algorithm,
                            dataset,
                            flightsYear = 2008,
                            ADMMmaxLocalIterations = maxLocalIterations,
                            ADMMlocalEpsilon = localEpsilon,
                            broadcastDelay = broadcastDelay)

            output = open(PICKLED_OUTPUT, 'wb')
            pickle.dump(results, output)
            output.close()

for runtime in RUNTIMES:
    for dim in [2, 100]:
        for skew in [0.0]:
            for algorithm in ALGORITHMS:
                broadcastDelay = -1
                localEpsilon = GLOBAL_ADMMlocalEpsilon
                if algorithm == "ADMM":
                    maxLocalIterations = GLOBAL_ADMM_maxLocalIterations
                elif algorithm == "MiniBatchADMM":
                    maxLocalIterations = GLOBAL_MiniBatchADMM_maxLocalIterations
                    localEpsilon = GLOBAL_MiniBatchADMM_localEpsilon
                elif algorithm == "HOGWILD":
                    maxLocalIterations = GLOBAL_HOGWILD_maxLocalIterations
                    broadcastDelay = GLOBAL_HOGWILD_broadcastDelay
                else:
                    maxLocalIterations = GLOBAL_AsyncADMM_maxLocalIterations
                    broadcastDelay = GLOBAL_AsyncADMM_broadcastDelay

                results += runTest(runtime,
                                   algorithm,
                                   "cloud",
                                   cloudPartitionSkew = skew,
                                   cloudDim = dim,
                                   ADMMmaxLocalIterations = maxLocalIterations,
                                   ADMMlocalEpsilon = localEpsilon,
                                   broadcastDelay = broadcastDelay)

                output = open(PICKLED_OUTPUT, 'wb')
                pickle.dump(results, output)
                output.close()

for runtime in RUNTIMES:
    for dim in [10]:
        for skew in [0.0, 0.1]:
            for algorithm in ALGORITHMS:
                broadcastDelay = -1
                localEpsilon = GLOBAL_ADMMlocalEpsilon
                if algorithm == "ADMM":
                    maxLocalIterations = GLOBAL_ADMM_maxLocalIterations
                elif algorithm == "MiniBatchADMM":
                    maxLocalIterations = GLOBAL_MiniBatchADMM_maxLocalIterations
                    localEpsilon = GLOBAL_MiniBatchADMM_localEpsilon
                elif algorithm == "HOGWILD":
                    maxLocalIterations = GLOBAL_HOGWILD_maxLocalIterations
                    broadcastDelay = GLOBAL_HOGWILD_broadcastDelay                    
                else:
                    maxLocalIterations = GLOBAL_AsyncADMM_maxLocalIterations
                    broadcastDelay = GLOBAL_AsyncADMM_broadcastDelay

                results += runTest(runtime,
                                   algorithm,
                                   "cloud",
                                   cloudPartitionSkew = skew,
                                   cloudDim = dim,
                                   ADMMmaxLocalIterations = maxLocalIterations,
                                   ADMMlocalEpsilon = localEpsilon,
                                   broadcastDelay = broadcastDelay)

                output = open(PICKLED_OUTPUT, 'wb')
                pickle.dump(results, output)
                output.close()


## END OF EXPERIMENT RUNS

# display the results
print results[0].keys()
for r in results:
    print [r[k] for k in r if k is not "line"]
    
