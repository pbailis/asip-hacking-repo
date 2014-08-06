
from os import system
from sys import exit
import pickle

ALGORITHMS = ["SVMADMM", "SVMADMMAsync", "HOGWILDSVM", "SVM"]

MASTER = "ec2-54-184-179-190.us-west-2.compute.amazonaws.com"


RUNTIMES = [1000, 5000, 10000, 20000]

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


def make_run_cmd(runtimeMS,
                 algorithm,
                 datasetConfigName,
                 datasetConfigStr,
                 regType="L2",
                 regParam=0.0001,
                 numPartitions = 40,
                 miscStr = ""):
    return "cd /mnt/spark; sbin/stop-all.sh; sleep 5; sbin/start-all.sh; sleep 3;" \
           "./bin/spark-submit " \
           "--driver-memory 52g " \
           "--class org.apache.spark.examples.mllib.research.SynchronousADMMTests " \
           "examples/target/scala-*/spark-examples-*.jar " \
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

def runTest(algorithm, cmd, dim=-1, skew=-1):
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



for runtime in RUNTIMES:
    for dim in [2, 100]:
        for skew in [0.0]:
            for algorithm in ALGORITHMS:
                if algorithm == "SVMADMM":
                    maxLocalIterations = 1000000000
                    broadcastDelay = 1000
                elif algorithm == "HOGWILDSVM":
                    maxLocalIterations = 100
                    broadcastDelay = 5
                else:
                    maxLocalIterations = 1000
                    broadcastDelay = 100
                dataset = describe_point_cloud(partitionSkew=skew, dimension=dim)
                results += runTest(algorithm, make_run_cmd(runtime, algorithm, "cloud", dataset,
                                                           miscStr="--ADMMmaxLocalIterations %d --ADMMepsilon 0.000001  --ADMMLocalepsilon 0.000001 --ADMMrho 1.0 --ADMMLagrangianrho 0.5 --broadcastDelayMs %d" % (maxLocalIterations, broadcastDelay)), skew=skew)
                # Pickel the output
                output = open('experiment.pkl', 'wb')
                pickle.dump(results, output)
                output.close()

for runtime in RUNTIMES:
    for algorithm in ALGORITHMS:
        dataset = describe_forest(MASTER)
        if algorithm == "SVMADMM":
            maxLocalIterations = 1000000000
            broadcastDelay = 1000
        elif algorithm == "HOGWILDSVM":
            maxLocalIterations = 100
            broadcastDelay = 5
        else:
            maxLocalIterations = 1000
            broadcastDelay = 100
        results += runTest(algorithm, make_run_cmd(runtime, algorithm, "bismarck", dataset,
                                                   miscStr="--ADMMmaxLocalIterations %d --ADMMepsilon 0.000001  --ADMMLocalepsilon 0.000001 --ADMMrho 1.0 --ADMMLagrangianrho 0.5 --broadcastDelayMs %d" % (maxLocalIterations, broadcastDelay)), skew=skew)
                # Pickel the output
        output = open('experiment.pkl', 'wb')
        pickle.dump(results, output)
        output.close()


for runtime in RUNTIMES:
    for algorithm in ALGORITHMS:
        dataset = describe_flights(MASTER, 2008)
        if algorithm == "SVMADMM":
            maxLocalIterations = 1000000000
            broadcastDelay = 1000
        elif algorithm == "HOGWILDSVM":
            maxLocalIterations = 100
            broadcastDelay = 5
        else:
            maxLocalIterations = 1000
            broadcastDelay = 100
        results += runTest(algorithm, make_run_cmd(runtime, algorithm, "bismarck", dataset,
                                                   miscStr="--ADMMmaxLocalIterations %d --ADMMepsilon 0.000001  --ADMMLocalepsilon 0.000001 --ADMMrho 1.0 --ADMMLagrangianrho 0.5 --broadcastDelayMs %d" % (maxLocalIterations, broadcastDelay)), skew=skew)
                # Pickel the output
        output = open('experiment.pkl', 'wb')
        pickle.dump(results, output)
        output.close()


#exit(0)

#results = []
for runtime in RUNTIMES:
    for dim in [10]:
        for skew in [0.0, 0.1]:
            for algorithm in ALGORITHMS:
                if algorithm == "SVMADMM":
                    maxLocalIterations = 1000000000
                    broadcastDelay = 1000
                elif algorithm == "HOGWILDSVM":
                    maxLocalIterations = 100
                    broadcastDelay = 5
                else:
                    maxLocalIterations = 1000
                    broadcastDelay = 100
                dataset = describe_point_cloud(partitionSkew=skew, dimension=dim)
                results += runTest(algorithm, make_run_cmd(runtime, algorithm, "cloud", dataset,
                                                           miscStr="--ADMMmaxLocalIterations %d --ADMMepsilon 0.000001 --ADMMLocalepsilon 0.000001 --ADMMrho 1.0 --ADMMLagrangianrho 0.5 --broadcastDelayMs %d" % (maxLocalIterations, broadcastDelay)), skew=skew)
                # Pickel the output
                output = open('experiment.pkl', 'wb')
                pickle.dump(results, output)
                output.close()



# display the results
print results[0].keys()
for r in results:
    print [r[k] for k in r if k is not "line"]
    
