
from os import system
from sys import exit
import pickle

## START OF EXPERIMENTAL PARAMETERS

RUNTIMES = [1000, 5000, 10000, 20000]

ALGORITHMS = ["SVMADMM", "SVMADMMAsync", "HOGWILDSVM", "SVM"]

HDFS_MASTER = "ec2-54-184-179-190.us-west-2.compute.amazonaws.com"

PICKLED_OUTPUT = "experiment.pkl"

## END OF EXPERIMENTAL PARAMETERS


## START OF CONSTANTS

GLOBAL_ADMMepsilon = 0.000001
GLOBAL_ADMMlocalEpsilon = 0.000001
GLOBAL_ADMMrho = 1.0
GLOBAL_ADMMlagrangianRho = 0.5

GLOBAL_SVMADMM_maxLocalIterations = 1000000000

GLOBAL_HOGWILDSVM_maxLocalIterations = 100
GLOBAL_HOGWILDSVM_broadcastDelay = 5

GLOBAL_SVMADMMAsync_maxLocalIterations = 1000
GLOBAL_SVMADMMAsync_broadcastDelay = 100

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

def describe_forest(master):
    return " --input hdfs://"+master+":9000/user/root/bismarck_data/forest* "

def describe_flights(master, year):
    return " --input hdfs://"+master+":9000/user/root/flights"+str(year)+".csv"

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
            numPartitions = 40,
            broadcastDelay = 100,
            cloudDim=-1,
            cloudPartitionSkew=-1,
            flightsYear = "2008",
            miscStr = ""):
    if datasetName == "forest":
        datasetConfigStr = describe_forest(HDFS_MASTER)
    elif datasetName == "cloud":
        datasetConfigStr = describe_point_cloud(partitionSkew = cloudPartitionSkew, dimension = cloudDim)
    elif datasetName == "flights":
        datasetConfigStr = describe_flights(HDFS_MASTER, flightsYear)
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
                "iterations": int(line[2]),
                "expected_runtime": int(line[3]),
                "runtime_ms": int(line[4]),
                "training_error": float(line[5]),
                "training_loss": float(line[6]),
                "reg_penalty": float(line[7]),
                "total_loss": line[8],
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
                "pointCloudDim": dim,
                "pointCloudSkew": skew
            }
            results.append(record)

    return results

results = []

## END OF TEST RUNNING CODE


## START OF EXPERIMENT RUNS

for runtime in RUNTIMES:
    for dim in [2, 100]:
        for skew in [0.0]:
            for algorithm in ALGORITHMS:
                broadcastDelay = -1
                if algorithm == "SVMADMM":
                    maxLocalIterations = GLOBAL_SVMADMM_maxLocalIterations
                elif algorithm == "HOGWILDSVM":
                    maxLocalIterations = GLOBAL_HOGWILDSVM_maxLocalIterations
                    broadcastDelay = GLOBAL_HOGWILDSVM_broadcastDelay
                else:
                    maxLocalIterations = GLOBAL_SVMADMMAsync_maxLocalIterations
                    broadcastDelay = GLOBAL_SVMADMMAsync_broadcastDelay

                results += runTest(runtime,
                                   algorithm,
                                   "cloud",
                                   cloudPartitionSkew = skew,
                                   cloudDim = dim,
                                   broadcastDelay = broadcastDelay)

                output = open(PICKLED_OUTPUT, 'wb')
                pickle.dump(results, output)
                output.close()

for runtime in RUNTIMES:
    for dim in [10]:
        for skew in [0.0, 0.1]:
            for algorithm in ALGORITHMS:
                broadcastDelay = -1
                if algorithm == "SVMADMM":
                    maxLocalIterations = GLOBAL_SVMADMM_maxLocalIterations
                elif algorithm == "HOGWILDSVM":
                    maxLocalIterations = GLOBAL_HOGWILDSVM_maxLocalIterations
                    broadcastDelay = GLOBAL_HOGWILDSVM_broadcastDelay
                else:
                    maxLocalIterations = GLOBAL_SVMADMMAsync_maxLocalIterations
                    broadcastDelay = GLOBAL_SVMADMMAsync_broadcastDelay

                results += runTest(runtime,
                                   algorithm,
                                   "cloud",
                                   cloudPartitionSkew = skew,
                                   cloudDim = dim,
                                   broadcastDelay = broadcastDelay)

                output = open(PICKLED_OUTPUT, 'wb')
                pickle.dump(results, output)
                output.close()

for runtime in RUNTIMES:
    for algorithm in ALGORITHMS:
        broadcastDelay = -1
        if algorithm == "SVMADMM":
            maxLocalIterations = GLOBAL_SVMADMM_maxLocalIterations
        elif algorithm == "HOGWILDSVM":
            maxLocalIterations = GLOBAL_HOGWILDSVM_maxLocalIterations
            broadcastDelay = GLOBAL_HOGWILDSVM_broadcastDelay
        else:
            maxLocalIterations = GLOBAL_SVMADMMAsync_maxLocalIterations
            broadcastDelay = GLOBAL_SVMADMMAsync_broadcastDelay

        results += runTest(runtime,
                           algorithm,
                           "forest",
                           broadcastDelay = broadcastDelay)

        output = open(PICKLED_OUTPUT, 'wb')
        pickle.dump(results, output)
        output.close()


for runtime in RUNTIMES:
    for algorithm in ALGORITHMS:
        broadcastDelay = -1
        if algorithm == "SVMADMM":
            maxLocalIterations = GLOBAL_SVMADMM_maxLocalIterations
        elif algorithm == "HOGWILDSVM":
            maxLocalIterations = GLOBAL_HOGWILDSVM_maxLocalIterations
            broadcastDelay = GLOBAL_HOGWILDSVM_broadcastDelay
        else:
            maxLocalIterations = GLOBAL_SVMADMMAsync_maxLocalIterations
            broadcastDelay = GLOBAL_SVMADMMAsync_broadcastDelay

        results += runTest(runtime,
                           algorithm,
                           "flights",
                           flightsYear=2008,
                           broadcastDelay = broadcastDelay)

        output = open(PICKLED_OUTPUT, 'wb')
        pickle.dump(results, output)
        output.close()

## END OF EXPERIMENT RUNS

# display the results
print results[0].keys()
for r in results:
    print [r[k] for k in r if k is not "line"]
    
