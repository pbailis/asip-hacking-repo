
from os import system

ALGORITHMS = ["SVM", "SVMADMM", "SVMADMMAsync"]

# class Result:
#     def __init__(self, algorithm, runtime_ms, area_under_pr, training_loss):
#         self.algorithm = algorithm
#         self.runtime_ms = runtime_ms
#         self.area_under_pr = area_under_pr
#         self.training_loss = training_loss

def describe_point_cloud(pointsPerPartition = 10000,
                         partitionSkew = 0,
                         labelNoise = 0.,
                         dimension = 3):
    return   "--pointCloudPointsPerPartition %d " \
             "--pointCloudPartitionSkew %f " \
             "--pointCloudLabelNoise %f " \
             "--pointCloudDimension %d " % \
             (pointsPerPartition,
              partitionSkew,
              labelNoise,
              dimension)

def make_run_cmd(algorithm,
                 datasetConfigName,
                 datasetConfigStr,
                 regType="L2",
                 regParam=0.0001,
                 numPartitions = 40,
                 iterationStart = 1,
                 iterationEnd = 12,
                 iterationStep = 2,
                 miscStr = ""):
    return "cd ~/spark; sbin/stop-all.sh; sleep 5; sbin/start-all.sh;" \
           "./bin/spark-submit " \
           "--class org.apache.spark.examples.mllib.research.SynchronousADMMTests" \
           " examples/target/scala-*/spark-examples-*.jar " \
           "--algorithm %s " \
           "--regType %s " \
           "--regParam %f " \
           "--format %s " \
           "--numPartitions %d " \
           "--sweepIterationStart %d " \
           "--sweepIterationEnd %d " \
           " --sweepIterationStep %d " \
           " %s %s " % \
            (algorithm,
             regType,
             regParam,
             datasetConfigName,
             numPartitions,
             iterationStart,
             iterationEnd,
             iterationStep,
             datasetConfigStr,
             miscStr)

def runTest(algorithm, cmd):
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
                "runtime_ms": int(line[3]),
                "pr": float(line[4]),
                "roc": float(line[5]),
                "training_loss": float(line[6]),
                "reg_penalty": float(line[7]),
                "total_loss": line[8],
                "model": line[9],
                "line": line
            }
            results.append(record)

    return results

for algorithm in ALGORITHMS:
    results = []
    results += runTest(algorithm, make_run_cmd(algorithm, "cloud", describe_point_cloud()))
    for r in results:
        print r
    
