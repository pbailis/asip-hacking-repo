package org.apache.spark.examples.mllib.research

import java.util.concurrent.TimeUnit

import org.apache.spark.examples.mllib.research.SynchronousADMMTests.Params
import org.apache.spark.mllib.classification.{SVMWithADMM, SVMWithAsyncADMM, SVMWithSGD}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.util.Random




object DataLoaders {
  def loadBismark(sc: SparkContext, filename: String): RDD[LabeledPoint] = {
    val data = sc.textFile(filename)
      .filter(s => !s.isEmpty && s(0) == '{')
      .map(s => s.split('\t'))
      .map {
      case Array(x, y) =>
        val features = x.stripPrefix("{").stripSuffix("}").split(',').map(xi => xi.toDouble)
        val label = if (y.toDouble > 0) 1 else 0
        LabeledPoint(label, new DenseVector(features))
    }.cache()

    data
  }

  def makeDictionary(colId: Int, tbl: RDD[Array[String]]): Map[String, Int] = {
    tbl.map(row => row(colId)).distinct.collect.zipWithIndex.toMap
  }

  def makeBinary(value: String, dict: Map[String, Int]): Array[Double] = {
    val array = new Array[Double](dict.size)
    array(dict(value)) = 1.0
    array
  }

  def loadFlights(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    val labels = Array("Year", "Month", "DayOfMonth", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime",
      "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime",
      "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiIn", "TaxiOut",
      "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay",
      "NASDelay", "SecurityDelay", "LateAircraftDelay").zipWithIndex.toMap
    println("Loading data")
    val rawData = sc.textFile(filename, params.numPartitions).
      filter(s => !s.contains("Year")).
      map(s => s.split(",")).cache()

    val carrierDict = makeDictionary(labels("UniqueCarrier"), rawData)
    val flightNumDict = makeDictionary(labels("FlightNum"), rawData)
    val tailNumDict = makeDictionary(labels("TailNum"), rawData)
    val originDict = makeDictionary(labels("Origin"), rawData)
    val destDict = makeDictionary(labels("Dest"), rawData)

    val data = rawData.map {
      row =>
        val firstFiveFeatures = (row.view(0, 5) ++ row.view(6, 7)).map {
          x =>
            if (x == "NA") 0.0 else x.toDouble
        }
        val carrierFeatures = makeBinary(row(labels("UniqueCarrier")), carrierDict)
        val flightFeatures = makeBinary(row(labels("FlightNum")), flightNumDict)
        val tailNumFeatures = makeBinary(row(labels("TailNum")), tailNumDict)
        val originFeatures = makeBinary(row(labels("Origin")), originDict)
        val destFeatures = makeBinary(row(labels("Dest")), destDict)
        val features: Array[Double] = (firstFiveFeatures ++ carrierFeatures ++ flightFeatures ++
          tailNumFeatures ++ originFeatures ++ destFeatures).toArray
        val delay = row(labels("ArrDelay"))
        val label = if (delay != "NA" && delay.toDouble > 0) 1.0 else 0.0

        val numNonZero = features.map( f => if(f > 0) 1 else 0).reduce(_ + _)

        val featureIndexes = new Array[Int](numNonZero)
        val featureValues = new Array[Double](numNonZero)
        var currentSparseIndex = 0

        for(i <- 0 until features.size) {
          val fv = features(i)
          if(fv > 0) {
            featureIndexes(currentSparseIndex) = i
            featureValues(currentSparseIndex) = fv
            currentSparseIndex += 1
          }
        }

        LabeledPoint(label, new SparseVector(numNonZero, featureIndexes, featureValues))
    }.cache()
    data.count
    println(s"THIS MANY PLUSES SUCKA ${data.filter(x => x.label == 1).count / data.count.toDouble}")
    data
  }

  /*
    Build a dataset of points drawn from one of two 'point clouds' (one at [5,...] and one at [10, ...])
    with either all positive or all negative labels.

    labelNoise controls how many points within each cloud are mislabeled
    cloudSize controls the radius of each cloud
    partitionSkew controls how much of each cloud is visible to each partition
      partitionSkew = 0 means each partition only sees one cloud
      partitionSkew = .5 means each partition sees half of each cloud
   */
  def generatePairCloud(sc: SparkContext,
                        dim: Int,
                        labelNoise: Double,
                        cloudSize: Double,
                        partitionSkew: Double,
                        numPartitions: Int,
                        pointsPerPartition: Int): RDD[LabeledPoint] = {
    sc.parallelize(1 to numPartitions, numPartitions).flatMap {
      idx =>
        val plusCloud = new DenseVector(Array.fill[Double](dim)(0.0))
        plusCloud.values(dim - 1) = 1
        val negCloud = new DenseVector(Array.fill[Double](dim)(5.0))
        negCloud.values(dim - 1) = 1

        // Seed the generator with the partition index
        val random = new Random(idx)

        val ret = new Array[LabeledPoint](pointsPerPartition)
        val isPartitionPlus = idx % 2 == 1

        for (pt <- 0 until pointsPerPartition) {
          val isPointPlus = if (random.nextDouble() < partitionSkew) isPartitionPlus else !isPartitionPlus
          val trueLabel: Double = if (isPointPlus) 1.0 else 0.0

          val pointCenter = if (isPointPlus) plusCloud else negCloud

          // calculate the actual point in the cloud
          val chosenPoint = new DenseVector(new Array[Double](dim))
          for (d <- 0 until dim - 1) {
            chosenPoint.values(d) = pointCenter.values(d) + random.nextGaussian() * cloudSize
          }
          chosenPoint.values(dim - 1) = 1.0

          val chosenLabel = if (random.nextDouble() < labelNoise) (trueLabel+1) % 2 else trueLabel

          ret(pt) = new LabeledPoint(chosenLabel, chosenPoint)
        }
        ret.iterator
      }
    }
  }

object SynchronousADMMTests {

  object Algorithm extends Enumeration {
    type Algorithm = Value
    val SVM, SVMADMM, SVMADMMAsync = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import org.apache.spark.examples.mllib.research.SynchronousADMMTests.Algorithm._
  import org.apache.spark.examples.mllib.research.SynchronousADMMTests.RegType._

  case class Params(
                     input: String = null,
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     algorithm: Algorithm = SVM,
                     regType: RegType = L2,
                     regParam: Double = 0.1,

                     ADMMepsilon: Double = 1.0e-5,
                     ADMMLocalepsilon: Double = 1.0e-5,
                     ADMMmaxLocalIterations: Int = Int.MaxValue,
                     localStats: Boolean = false,

                     format: String = "libsvm",
                     numPartitions: Int = 128,
                     sweepIterationStart: Int = -1,
                     sweepIterationEnd: Int = -1,
                     sweepIterationStep: Int = -1,
                     pointCloudDimension: Int = 10,
                     pointCloudLabelNoise: Double = .2,
                     pointCloudPartitionSkew: Double = 0,
                     pointCloudPointsPerPartition: Int = 10000,
                     pointCloudSize: Double = 1.0)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("BinaryClassification") {
      head("BinaryClassification: an example app for binary classification.")

      // run a one-off test
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))

      // run a set of iterations
      opt[Int]("sweepIterationStart")
        .action((x, c) => c.copy(sweepIterationStart = x))
      opt[Int]("sweepIterationEnd")
        .action((x, c) => c.copy(sweepIterationEnd = x))
      opt[Int]("sweepIterationStep")
        .action((x, c) => c.copy(sweepIterationStep = x))

      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))


      // point cloud parameters
      opt[Int]("pointCloudDimension")
        .action((x, c) => c.copy(pointCloudDimension = x))
      opt[Double]("pointCloudLabelNoise")
        .action((x, c) => c.copy(pointCloudLabelNoise = x))
      opt[Double]("pointCloudPartitionSkew")
        .action((x, c) => c.copy(pointCloudPartitionSkew = x))
      opt[Int]("pointCloudPointsPerPartition")
        .action((x, c) => c.copy(pointCloudPointsPerPartition = x))
      opt[Double]("pointCloudRadius")
        .action((x, c) => c.copy(pointCloudSize = x))

      opt[String]("algorithm")
        .text(s"algorithm (${Algorithm.values.mkString(",")}), " +
        s"default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = Algorithm.withName(x)))
      opt[String]("regType")
        .text(s"regularization type (${RegType.values.mkString(",")}), " +
        s"default: ${defaultParams.regType}")
        .action((x, c) => c.copy(regType = RegType.withName(x)))
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
        .action((x, c) => c.copy(regParam = x))
      opt[Int]("numPartitions")
        .action((x, c) => c.copy(numPartitions = x))
      opt[String]("input")
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input = x))
      opt[String]("format")
        .text("File format")
        .action((x, c) => c.copy(format = x))

      // ADMM-specific stuff
      opt[Double]("ADMMepsilon")
        .action((x, c) => c.copy(ADMMepsilon = x))
      opt[Double]("ADMMLocalepsilon")
        .action((x, c) => c.copy(ADMMLocalepsilon = x))

      opt[Int]("ADMMmaxLocalIterations")
        .action((x, c) => c.copy(ADMMmaxLocalIterations = x))
      opt[Boolean]("localStats")
        .action((x, c) => c.copy(localStats = x))

      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.BinaryClassification \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --algorithm SVM --regType L2 --regParam 1.0 \
          |  data/mllib/sample_binary_classification_data.txt
        """.stripMargin)
    }



    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"BinaryClassification with $params")
    val sc = new SparkContext(conf)

    println("Starting to load data...")

    val examples = if (params.format == "lisbsvm") {
      MLUtils.loadLibSVMFile(sc, params.input).cache()
    } else if (params.format == "bismarck") {
      DataLoaders.loadBismark(sc, params.input).cache()
    } else if (params.format == "flights") {
      DataLoaders.loadFlights(sc, params.input, params).cache()
    } else if (params.format == "cloud") {
      DataLoaders.generatePairCloud(sc,
        params.pointCloudDimension,
        params.pointCloudLabelNoise,
        params.pointCloudSize,
        params.pointCloudPartitionSkew,
        params.numPartitions,
        params.pointCloudPointsPerPartition).cache()
    } else {
      throw new RuntimeException(s"Unrecognized input format ${params.format}")
    }



//    val splits = examples.randomSplit(Array(0.8, 0.2))
//    val training = splits(0).cache()
//    val test = splits(1).cache()
    val training = examples

//    {
//      val fos = new FileOutputStream("train.tsv")
//      val pw = new PrintWriter(fos)
//      training.collect().foreach(pt => pw.println(s"${pt.label}\t${pt.features.toArray.mkString("\t")}"))
//      pw.flush()
//      pw.close()
//    }
//    {
//      val fos = new FileOutputStream("test.tsv")
//      val pw = new PrintWriter(fos)
//      test.collect().foreach(pt => pw.println(s"${pt.label}\t${pt.features.toArray.mkString("\t")}"))
//      pw.flush()
//      pw.close()
//    }


    training.repartition(params.numPartitions).cache()
    //test.repartition(params.numPartitions)

    val numTraining = training.count()
    //    val numTest = test.count()

    println(s"Loaded data! Number of training examples: $numTraining")
    println(s"Number of partitions: ${training.partitions.length}")

    // examples.unpersist(blocking = false)

    val updater = params.regType match {
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    println("Starting test!")

    var it_st = params.sweepIterationStart
    var it_end = params.sweepIterationEnd
    var it_step = params.sweepIterationStep

    // this is terrible, but it's perhaps more of a venial sin
    if (it_st == -1) {
      it_st = params.numIterations
      it_end = it_st + 1
      it_step = 10
    }

    for (i <- it_st to it_end by it_step) {
      val startTime = System.nanoTime()

      val model = runTest(training, updater, params, i)

      val totalTimeNs = System.nanoTime() - startTime
      val totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

      val trainingError = training.map{ point =>
        val p = model.predict(point.features)
        val y = 2.0 * point.label - 1.0
        if (y * p <= 0.0) 1.0 else 0.0
      }.reduce(_ + _) / training.count.toDouble

      val trainingLoss = model.loss(training)
      val regularizationPenalty = params.regParam * math.pow(model.weights.l2Norm,2)


      println(s"Iterations = ${i}")
      println(s"Training error = ${trainingError}.")
      println(s"Training (Loss, reg, total) = ${trainingLoss}, ${regularizationPenalty}, ${trainingLoss + regularizationPenalty}")
      println(s"Total time ${totalTimeMs}ms")

      val summary =
        s"RESULT: ${params.algorithm}\t${i}\t${totalTimeMs}\t${trainingError}" +
        s"\t${trainingLoss}\t ${regularizationPenalty}\t${trainingLoss + regularizationPenalty}" +
        s"\t${model.weights.toArray.mkString(",")}"

      println(summary)

    }

    sc.stop()
  }

  def runTest(training: RDD[LabeledPoint],
              updater: Updater,
              params: Params,
              iterations: Int) = {
    println(s"Running algorithm ${params.algorithm} $iterations iterations")

    params.algorithm match {
//      case LR =>
//        val algorithm = new LogisticRegressionWithSGD()
//        algorithm.optimizer
//          .setNumIterations(iterations)
//          .setStepSize(params.stepSize)
//          .setUpdater(updater)
//          .setRegParam(params.regParam)
//        algorithm.run(training).clearThreshold()
      case SVM =>
        val algorithm = new SVMWithSGD()
        algorithm.optimizer
          .setNumIterations(iterations)
          .setStepSize(params.stepSize)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        val model = algorithm.run(training).clearThreshold()
        println(model.weights.toArray.mkString(","))
        println(model.intercept)
        model
      case SVMADMM =>
        val algorithm = new SVMWithADMM()
        algorithm.maxGlobalIterations = iterations
        algorithm.maxLocalIterations = params.ADMMmaxLocalIterations
        algorithm.regParam = params.regParam
        algorithm.epsilon = params.ADMMepsilon
        algorithm.localEpsilon = params.ADMMLocalepsilon
        algorithm.collectLocalStats = params.localStats
        algorithm.setup()
        algorithm.run(training).clearThreshold()
      case SVMADMMAsync =>
        val algorithm = new SVMWithAsyncADMM()
        algorithm.maxLocalIterations = params.ADMMmaxLocalIterations
        algorithm.regParam = params.regParam
        algorithm.epsilon = params.ADMMepsilon
        algorithm.broadcastDelayMS = 100
        algorithm.runtimeMS = iterations * 1000
        algorithm.setup()
        algorithm.run(training).clearThreshold()
    }
  }
}
