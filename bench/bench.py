
from os import system
import pickle



ALGORITHMS = ["SVM", "SVMADMM", "SVMADMMAsync"]

# class Result:
#     def __init__(self, algorithm, runtime_ms, area_under_pr, training_loss):
#         self.algorithm = algorithm
#         self.runtime_ms = runtime_ms
#         self.area_under_pr = area_under_pr
#         self.training_loss = training_loss

def describe_point_cloud(pointsPerPartition = 100000,
                         partitionSkew = 0.00,
                         labelNoise = 0.2,
                         dimension = 100):
    return   "--pointCloudPointsPerPartition %d " \
             "--pointCloudPartitionSkew %f " \
             "--pointCloudLabelNoise %f " \
             "--pointCloudDimension %d " % \
             (pointsPerPartition,
              partitionSkew,
              labelNoise,
              dimension)

def make_run_cmd(runtimeMS,
                 algorithm,
                 datasetConfigName,
                 datasetConfigStr,
                 regType="L2",
                 regParam=0.0001,
                 numPartitions = 40,
                 miscStr = ""):
    return "cd /mnt/spark; sbin/stop-all.sh; sleep 5; sbin/start-all.sh;" \
           "./bin/spark-submit " \
           "--class org.apache.spark.examples.mllib.research.SynchronousADMMTests" \
           " examples/target/scala-*/spark-examples-*.jar " \
           "--algorithm %s " \
           "--regType %s " \
           "--regParam %f " \
           "--format %s " \
           "--numPartitions %d " \
           "--runtimeMS %d" \
           " %s %s " % \
            (algorithm,
             regType,
             regParam,
             datasetConfigName,
             numPartitions,
             runtimeMS,
             datasetConfigStr,
             miscStr)

def runTest(algorithm, cmd, dim, skew):
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
                "line": line,
                "dim": dim,
                "skew": skew,
                "command": cmd
            }
            results.append(record)

    return results



results = []
for runtime in range(5, 50, 5):
    for dim in [3, 5, 10, 50, 100]:
        for skew in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for algorithm in ALGORITHMS:
                dataset = describe_point_cloud(partitionSkew=skew, dimension=dim)
                results += runTest(algorithm, make_run_cmd(runtime * 1000, algorithm, "cloud", dataset,
                                                       miscStr="--ADMMmaxLocalIterations 500"), dim, skew)
                # Pickel the output
                output = open('experiment.pkl', 'wb')
                pickle.dump(results, output)
                output.close()

# display the results
print results[0].keys()
for r in results:
    print [r[k] for k in r if k is not "line"]
    
