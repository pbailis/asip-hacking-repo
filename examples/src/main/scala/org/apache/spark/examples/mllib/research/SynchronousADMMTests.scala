package org.apache.spark.examples.mllib.research

import org.apache.log4j.{Level, Logger}
import org.apache.spark.examples.mllib.research.SynchronousADMMTests.Params
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.util.Random


object DataLoaders {
  def loadBismark(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    val data = sc.textFile(filename, params.numPartitions)
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
        val value_arr = Array.fill(12)(1.0)
        val idx_arr = new Array[Int](12)

        var idx_offset = 0
        for(i <- 0 until 7 if i != 5) {
          value_arr(idx_offset) = if (row(i) == "NA") 0.0 else row(i).toDouble
          idx_arr(idx_offset) = idx_offset
          idx_offset += 1
        }

        var bitvector_offset = idx_offset
        idx_arr(idx_offset) = bitvector_offset + carrierDict(row(labels("UniqueCarrier")))
        idx_offset += 1
        bitvector_offset += carrierDict.size

        idx_arr(idx_offset) = bitvector_offset + flightNumDict(row(labels("FlightNum")))
        idx_offset += 1
        bitvector_offset += flightNumDict.size

        idx_arr(idx_offset) = bitvector_offset + tailNumDict(row(labels("TailNum")))
        idx_offset += 1
        bitvector_offset += tailNumDict.size

        idx_arr(idx_offset) = bitvector_offset + originDict(row(labels("Origin")))
        idx_offset += 1
        bitvector_offset += originDict.size

        idx_arr(idx_offset) = bitvector_offset + destDict(row(labels("Dest")))
        idx_offset += 1
        bitvector_offset += destDict.size

        // add one for bias term
        bitvector_offset += 1

        val delay = row(labels("ArrDelay"))
        val label = if (delay != "NA" && delay.toDouble > 0) 1.0 else 0.0

        assert(idx_offset == 11)

        LabeledPoint(label, new SparseVector(bitvector_offset, idx_arr, value_arr))
    }

    val ret = data.repartition(params.numPartitions)
    ret.cache().count()
    rawData.unpersist(true)

    ret
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
    sc.parallelize(1 to numPartitions, numPartitions).flatMap { idx =>
      val plusCloud = new DenseVector(Array.fill[Double](dim)(10.0))
      plusCloud.values(dim - 1) = 1
      val negCloud = new DenseVector(Array.fill[Double](dim)(5.0))
      negCloud.values(dim - 1) = 1

      // Seed the generator with the partition index
      val random = new Random(idx)
      val isPartitionPlus = idx % 2 == 1

      (0 until pointsPerPartition).iterator.map { pt =>
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

        new LabeledPoint(chosenLabel, chosenPoint)
      }
    }
  }
}

object SynchronousADMMTests {

  object Algorithm extends Enumeration {
    type Algorithm = Value
    val SVM, SVMADMM, SVMADMMAsync, LR, LRADMM, LRADMMAsync, HOGWILDSVM = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import org.apache.spark.examples.mllib.research.SynchronousADMMTests.Algorithm._
  import org.apache.spark.examples.mllib.research.SynchronousADMMTests.RegType._

  class Params extends ADMMParams {
    var input: String = null
    var  format: String = "libsvm"
    var numPartitions: Int = -1
    var algorithm: Algorithm = SVM
    var regType: RegType = L2
    var pointCloudDimension: Int = 10
    var pointCloudLabelNoise: Double = .2
    var pointCloudPartitionSkew: Double = 0
    var pointCloudPointsPerPartition: Int = 10000
    var pointCloudSize: Double = 1.0

    override def toString = {
      "{" + "input: " + input + ", " +
      "format: " + format + ", " +
      "numPartitions: " + numPartitions + ", " +
      "algorithm: " + algorithm + ", " +
      "algParams: " + super.toString() + ", " +
      "regType: " + regType + ", " +
      "pointCloudDim: " + pointCloudDimension + ", " +
      "pointCloudNoise: " + pointCloudLabelNoise + ", " +
      "pointCloudSkew: " + pointCloudPartitionSkew + ", " +
      "pointCloudPoints: " + pointCloudPointsPerPartition + ", " +
      "pointCloudSize: " + pointCloudSize + "}"
    }
  }

  def main(args: Array[String]) {
    val defaultParams = new Params()

    Logger.getRootLogger.setLevel(Level.WARN)

    val parser = new OptionParser[Params]("BinaryClassification") {
      head("BinaryClassification: an example app for binary classification.")

      // run a one-off test
      opt[Int]("runtimeMS")
        .text("runtime in miliseconds")
        .action { (x, c) => c.runtimeMS = x; c }
      opt[Int]("iterations")
        .text(s"num iterations: default ${defaultParams.maxIterations}")
        .action { (x, c) => c.maxIterations = x; c }
      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.eta_0}")
        .action { (x, c) => c.eta_0 = x; c}


      // point cloud parameters
      opt[Int]("pointCloudDimension")
        .action { (x, c) => c.pointCloudDimension = x; c }
      opt[Double]("pointCloudLabelNoise")
        .action { (x, c) => c.pointCloudLabelNoise = x; c }
      opt[Double]("pointCloudPartitionSkew")
        .action { (x, c) => c.pointCloudPartitionSkew = x; c }
      opt[Int]("pointCloudPointsPerPartition")
        .action { (x, c) => c.pointCloudPointsPerPartition = x; c }
      opt[Double]("pointCloudRadius")
        .action { (x, c) => c.pointCloudSize = x; c }

      opt[String]("algorithm")
        .text(s"algorithm (${Algorithm.values.mkString(",")}), " +
        s"default: ${defaultParams.algorithm}")
        .action { (x, c) => c.algorithm = Algorithm.withName(x); c }
      opt[String]("regType")
        .text(s"regularization type (${RegType.values.mkString(",")}), " +
        s"default: ${defaultParams.regType}")
        .action { (x, c) => c.regType = RegType.withName(x); c }

      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
        .action { (x, c) => c.regParam = x; c }
      opt[Int]("numPartitions")
        .action { (x, c) => c.numPartitions = x; c }
      opt[String]("input")
        .text("input paths to labeled examples in LIBSVM format")
        .action { (x, c) => c.input = x; c }

      opt[Double]("ADMMrho")
        .action { (x, c) => c.rho0 = x; c }

      opt[Double]("ADMMLagrangianrho")
        .action { (x, c) => c.lagrangianRho = x; c }
      opt[String]("format")
        .text("File format")
        .action { (x, c) => c.format = x; c }

      // ADMM-specific stuff
      opt[Double]("ADMMepsilon")
        .action { (x, c) => c.tol = x; c }
      opt[Double]("ADMMLocalepsilon")
        .action { (x, c) => c.workerTol = x; c }
      opt[Int]("broadcastDelayMs")
        .action { (x, c) => c.broadcastDelayMS = x; c }
      opt[Int]("ADMMmaxLocalIterations")
        .action{ (x, c) => c.maxWorkerIterations =x; c }
      opt[Boolean]("useLBFGS")
        .action{ (x, c) => c.useLBFGS = x; c }
      opt[Boolean]("adpativeRho")
        .action{ (x, c) => c.adaptiveRho = x; c }
      opt[Boolean]("usePorkChop")
        .action{ (x, c) => c.usePorkChop = x; c }
      opt[Boolean]("localStats")
        .action{ (x, c) => c.displayIncrementalStats = x; c }


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
    println(params)

    val conf = new SparkConf().setAppName(s"BinaryClassification with $params")
    conf.set("spark.akka.threads", "16")

    val sc = new SparkContext(conf)

    println("Starting to load data...")

    val examples = if (params.format == "lisbsvm") {
      MLUtils.loadLibSVMFile(sc, params.input)
    } else if (params.format == "bismarck") {
      DataLoaders.loadBismark(sc, params.input, params)
    } else if (params.format == "flights") {
      DataLoaders.loadFlights(sc, params.input, params)
    } else if (params.format == "cloud") {
      DataLoaders.generatePairCloud(sc,
        params.pointCloudDimension,
        params.pointCloudLabelNoise,
        params.pointCloudSize,
        params.pointCloudPartitionSkew,
        params.numPartitions,
        params.pointCloudPointsPerPartition)
    } else {
      throw new RuntimeException(s"Unrecognized input format ${params.format}")
    }

//    val splits = examples.randomSplit(Array(0.8, 0.2))
//    val training = splits(0).cache()
//    val test = splits(1).cache()
    val training = examples

    training.cache()
    //test.repartition(params.numPartitions)

    val numTraining = training.count()
    //    val numTest = test.count()

    println(s"Loaded data! Number of training examples: $numTraining")
    println(s"Number of partitions: ${training.partitions.length}")


    println("Starting test!")

    val (model, stats) = runTest(training, params)

    val trainingError = training.map{ point =>
      val p = model.predict(point.features)
      val y = 2.0 * point.label - 1.0
      if (y * p <= 0.0) 1.0 else 0.0
    }.reduce(_ + _) / numTraining.toDouble

    val trainingLoss = model.loss(training)
    val regularizationPenalty = params.regParam * math.pow(model.weights.l2Norm,2)


    println(s"desiredRuntime = ${params.runtimeMS}")
    println(s"Training error = ${trainingError}.")
    println(s"Training (Loss, reg, total) = ${trainingLoss}, ${regularizationPenalty}, ${trainingLoss + regularizationPenalty}")
    println(s"Total time ${stats("runtime")}ms")

    val summary =
      s"RESULT: ${params.algorithm}\t" +
      s"${stats("iterations")}\t" +
      s"${params.runtimeMS}\t" +
      s"${stats("runtime")}\t" +
      s"$trainingError\t" +
      s"$trainingLoss\t" +
      s"$regularizationPenalty\t" +
      s"${trainingLoss + regularizationPenalty}\t" +
      s"${stats.toString}\t" +
      s"${model.weights.toArray.mkString(",")}\t" +
      s"${params.toString}\t"

    println(summary)


    sc.stop()
  }

  def runTest(training: RDD[LabeledPoint], params: Params):
  (GeneralizedLinearModel, Map[String, String]) = {
    println(s"Running algorithm ${params.algorithm} for ${params.runtimeMS} MS")

    val updater = params.regType match {
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val consensusFun = params.regType match {
      case L1 => new L1ConsensusFunction()
      case L2 => new L2ConsensusFunction()
    }

    params.algorithm match {
      case LR =>
        val algorithm = new LogisticRegressionWithSGD()
        algorithm.optimizer
          .setNumIterations(100000)
          .setRuntime(params.runtimeMS)
          .setStepSize(params.eta_0)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        val model = algorithm.run(training).clearThreshold()
        val results = 
          Map(
            "iterations" -> algorithm.optimizer.getLastIterations().toString,
            "runtime" -> algorithm.optimizer.totalTimeMs.toString
          )
        (model, results)
      case SVM =>
        val algorithm = new SVMWithSGD()
        algorithm.optimizer
          .setNumIterations(100000)
          .setRuntime(params.runtimeMS)
          .setStepSize(params.eta_0)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        val startTime = System.nanoTime()
        val model = algorithm.run(training).clearThreshold()
        val results =
          Map(
            "iterations" -> algorithm.optimizer.getLastIterations().toString,
            "runtime" -> algorithm.optimizer.totalTimeMs.toString
          )
        (model, results)
      case SVMADMM =>
        val algorithm = new SVMWithADMM(params)
        algorithm.optimizer.consensus = consensusFun
        val startTime = System.nanoTime()
        val model = algorithm.run(training).clearThreshold()
        val results =
          Map(
            "iterations" -> algorithm.optimizer.iteration.toString,
            "avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
            "runtime" -> algorithm.optimizer.totalTimeMs.toString
          )
        (model, results )
      case SVMADMMAsync =>
        val algorithm = new SVMWithAsyncADMM(params)
        algorithm.optimizer.consensus = consensusFun
        val model = algorithm.run(training).clearThreshold()
        val results =
          Map(
            "iterations" -> algorithm.optimizer.stats.avgLocalIters().toString,
            "avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
            "avgMsgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
            "runtime" -> algorithm.optimizer.totalTimeMs.toString
          )
        println(results)
        (model, results)
      case HOGWILDSVM =>
        val algorithm = new SVMWithHOGWILD(params)
        algorithm.optimizer.consensus = consensusFun
        val model = algorithm.run(training).clearThreshold()
        val results =
          Map(
            "iterations" -> algorithm.optimizer.stats.avgLocalIters().toString,
            "msgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
            "runtime" -> algorithm.optimizer.totalTimeMs.toString
          )
        (model, results)

      // case LRADMM =>
      //   val algorithm = new LRWithADMM()
      //   algorithm.consensus = consensusFun
      //   //        algorithm.maxGlobalIterations = iterations
      //   algorithm.maxLocalIterations = params.ADMMmaxLocalIterations
      //   algorithm.regParam = params.regParam
      //   algorithm.epsilon = params.ADMMepsilon
      //   algorithm.localEpsilon = params.ADMMLocalepsilon
      //   algorithm.collectLocalStats = params.localStats
      //   algorithm.runtimeMS = params.runtimeMS
      //   algorithm.setup()
      //   val startTime = System.nanoTime()
      //   val model = algorithm.run(training).clearThreshold()
      //   (model, algorithm.optimizer.iteration, algorithm.optimizer.totalTimeMs)
      // case LRADMMAsync =>
      //   val algorithm = new LRWithAsyncADMM()
      //   algorithm.consensus = consensusFun
      //   algorithm.maxLocalIterations = params.ADMMmaxLocalIterations
      //   algorithm.regParam = params.regParam
      //   algorithm.epsilon = params.ADMMepsilon
      //   algorithm.localEpsilon = params.ADMMLocalepsilon
      //   algorithm.broadcastDelayMS = 100
      //   algorithm.runtimeMS = params.runtimeMS
      //   algorithm.rho = params.rho
      //   algorithm.lagrangianRho = params.lagrangianRho
      //   algorithm.setup()
      //   val model = algorithm.run(training).clearThreshold()
      //   (model, algorithm.optimizer.commStages, algorithm.optimizer.totalTimeMs)

    }
  }
}
