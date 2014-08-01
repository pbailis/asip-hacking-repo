
from os import system

ALGORITHMS = ["LR", "SVM", "SVMADMM", "SVMADMMAsync"]

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
    return "cd spark; sbin/stop-all.sh; sleep 5; sbin/start-all.sh;" \
           "./bin/spark-submit " \
           "--class org.apache.spark.examples.mllib.research.SynchronousADMMTests" \
           " examples/target/scala-*/spark-examples-*.jar " \
           "--algorithm %s " \
           "--regType %s " \
           "--regParam %f " \
           "--format %s |" \
           "--numPartitions %d " \
           "--sweepIterationStart %d " \
           "--sweepIterationEnd %d " \
           " --sweepIterationStep %d " \
           " %s %s " \
           " | grep RESULT | cut -c 9-" % \
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

def runTest(cmd):
    system("%s 2>&1 > /tmp/run.out" % (cmd))

    for line in open("/tmp/run.out"):
        print line

for alg in ALGORITHMS:
    runCmd = make_run_cmd(alg, "cloud", describe_point_cloud())