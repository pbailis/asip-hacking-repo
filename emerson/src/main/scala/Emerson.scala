package edu.berkeley.emerson

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser


object Emerson {

  object Algorithm extends Enumeration {
    type Algorithm = Value
    val GD, ADMM, MiniBatchADMM, AsyncADMM, PORKCHOP, HOGWILD, DualDecomp = Value
  }

  object Objective extends Enumeration {
    type Objective = Value
    val SVM, Logistic = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import edu.berkeley.emerson.Emerson.Algorithm._
  import edu.berkeley.emerson.Emerson.Objective._
  import edu.berkeley.emerson.Emerson.RegType._

  def mapToJson(m: Map[String, Any]): String = {
    "{" + m.iterator.map {
      case (k,v) => "\"" + k + "\": " + v
    }.toArray.mkString(", ") + "}"
  }

  class Params extends EmersonParams {
    var input: String = null
    var format: String = "libsvm"
    var numPartitions: Int = -1
    var algorithm: Algorithm = ADMM
    var objectiveFn: Objective = SVM
    var regType: RegType = L2
    var pointCloudDimension: Int = 10
    var pointCloudLabelNoise: Double = .2
    var pointCloudPartitionSkew: Double = 0
    var pointCloudPointsPerPartition: Int = 10000
    var pointCloudSize: Double = 1.0
    var inputTokenHashKernelDimension: Int = 100
    var dblpSplitYear = 2007
    // 4690 == database
    var wikipediaTargetWordToken = 4690
    var numTraining = 0L
    var scaled = false

    override def toString = {
      val m =  Map(
        "input" -> ("\"" + input + "\""),
        "format" -> ("\"" + format + "\""),
        "numPartitions" -> numPartitions,
        "algorithm" -> ("\"" + algorithm + "\""),
        "objective" -> ("\"" + objectiveFn + "\""),
        "algParams" -> super.toString(),
        "regType" -> ("\"" + regType + "\""),
        "pointCloudDim" -> pointCloudDimension,
        "pointCloudNoise" -> pointCloudLabelNoise,
        "pointCloudSkew" -> pointCloudPartitionSkew,
        "pointCloudPoints" -> pointCloudPointsPerPartition,
        "pointCloudSize" -> pointCloudSize,
        "inputTokenHashKernelDimension" -> inputTokenHashKernelDimension,
        "wikipediaTargetWordToken" -> wikipediaTargetWordToken,
        "dblpSplitYear" -> dblpSplitYear,
        "scaled" -> scaled,
        "numTraining" -> numTraining        
      )
      mapToJson(m)
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

      opt[Boolean]("learningT")
        .action{ (x, c) => c.learningT = x; c }

      // point cloud parameters
      opt[Int]("pointCloudDimension")
        .action { (x, c) => c.pointCloudDimension = x; c }
      opt[Double]("pointCloudLabelNoise")
        .action { (x, c) => c.pointCloudLabelNoise = x; c }
      opt[Double]("pointCloudPartitionSkew")
        .action { (x, c) => c.pointCloudPartitionSkew = x; c }
      opt[Int]("pointCloudPointsPerPartition")
        .action { (x, c) => c.pointCloudPointsPerPartition = x; c }

      opt[Int]("localTimeout")
      .action { (x, c) => c.localTimeout = x; c}

      opt[Double]("pointCloudRadius")
        .action { (x, c) => c.pointCloudSize = x; c }

      opt[Int]("inputTokenHashKernelDimension")
        .text("Used to downsample the input space for several datasets (DBLP, Wikipedia); -1 means do not downsample")
        .action { (x, c) => c.inputTokenHashKernelDimension = x; c }

      opt[Int]("dblpSplitYear")
        .text("In DBLP dataset, years less than this will be negatively labeled")
        .action { (x, c) => c.dblpSplitYear = x; c }

      opt[Int]("wikipediaTargetWordToken")
        .text("token word to try to predict in wikipedia dataset")
        .action { (x, c) => c.wikipediaTargetWordToken; c }

      opt[String]("algorithm")
        .text(s"algorithm (${Algorithm.values.mkString(",")}), " +
        s"default: ${defaultParams.algorithm}")
        .action { (x, c) => c.algorithm = Algorithm.withName(x); c }
      opt[String]("objective")
        .text(s"objective (${Objective.values.mkString(",")}), " +
        s"default: ${defaultParams.objectiveFn}")
        .action { (x, c) => c.objectiveFn = Objective.withName(x); c }
      opt[String]("regType")
        .text(s"regularization type (${RegType.values.mkString(",")}), " +
        s"default: ${defaultParams.regType}")
        .action { (x, c) => c.regType = RegType.withName(x); c }

      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
        .action { (x, c) => c.regParam = x; c }

      // opt[Double]("admmRegFactor")
      //   .text(s"scales the regularization parameter for ADMM: ${defaultParams.admmRegFactor}")
      //   .action { (x, c) => c.admmRegFactor = x; c }


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
        .action{ (x, c) => c.maxWorkerIterations = x; c }
      opt[Int]("miniBatchSize")
        .action{ (x, c) => c.miniBatchSize = x; c }
      opt[Boolean]("useLBFGS")
        .action{ (x, c) => c.useLBFGS = x; c }
      opt[Boolean]("adpativeRho")
        .action{ (x, c) => c.adaptiveRho = x; c }
      opt[Boolean]("usePorkChop")
        .action{ (x, c) => c.usePorkChop = x; c }
      opt[Boolean]("useLineSearch")
        .action{ (x, c) => c.useLineSearch = x; c }
      opt[Boolean]("localStats")
        .action{ (x, c) => c.displayIncrementalStats = x; c }

      // Scale the constants by N so that they can be all set to 0.01 or something robust
      // don't use this yet!
      opt[Boolean]("scaled")
        .action { (x, c) => c.scaled = x; c }


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
      params => run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    println(params)

    if (params.algorithm == PORKCHOP || params.usePorkChop == true) {
      // force the use of porkchop
      params.usePorkChop = true
      params.algorithm = PORKCHOP
    }


    val conf = new SparkConf().setAppName(s"BinaryClassification with $params")
    conf.set("spark.akka.threads", "16")
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.akka.heartbeat.interval", "20000") // in seconds?
    conf.set("spark.locality.wait", "100000") // in milliseconds

    val sc = new SparkContext(conf)

    println("Starting to load data...")

    var training: RDD[Array[(Double, BV[Double])]] =
      if (params.format == "lisbsvm") {
        MLUtils.loadLibSVMFile(sc, params.input).map(
          p => (p.label, p.features.toBreeze)
        ).repartition(params.numPartitions).mapPartitions( iter => Iterator(iter.toArray) ).cache()
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
      } else if (params.format == "dblp") {
        DataLoaders.loadDBLP(sc, params.input, params)
      } else if (params.format == "wikipedia") {
        DataLoaders.loadWikipedia(sc, params.input, params)
      } else {
        throw new RuntimeException(s"Unrecognized input format ${params.format}")
      }

    // temporarily disabled
    if (params.format == "dblp" || params.format == "wikipedia") {
      training = DataLoaders.normalizeData(training)
    }

  
    training.foreach{ x => System.gc() }
    val numTraining = training.map{x => x.length}.reduce(_ + _)
    params.numTraining = numTraining

    println(s"Loaded data! Number of training examples: $numTraining")
    println(s"Number of partitions: ${training.partitions.length}")


    println("Starting test!")

    val lossFunction = params.objectiveFn match {
      case SVM => new HingeLoss()
      case Logistic => new LogisticLoss()
    }

    val regularizationFunction = params.regType match {
      case L1 => new L1Regularizer()
      case L2 => new L2Regularizer()
    }

    val optimizer: EmersonOptimizer = params.algorithm match {
      case ADMM => new ADMM()
      case AsyncADMM | PORKCHOP => new AsyncADMM()
      case HOGWILD => new HOGWILDSGD()
      case DualDecomp => new DualDecomp()
    }

    val model = new EmersonModel(params, lossFunction, regularizationFunction, optimizer)


    val nDim = training.map(d => d(0)._2.size).take(1).head
    val initialWeights = BDV.zeros[Double](nDim)

    params.eta_0 *= nDim * 0.02

    model.fit(params, initialWeights, training)

    // Evaluate the model
    val (objective, propError, loss, regPenalty) = model.score(training)

    // Get any stats from the model
    val optimizerStats: Map[String, String] = model.optimizer.statsMap()

//    val trainingError = params.objective match {
//      case Logistic =>
//        training.map{ point =>
//          val p = model.predict(point.features)
//          if ( (p > 0.5) != (point.label > 0) ) 1.0 else 0.0
//        }.reduce(_ + _) / numTraining.toDouble
//      case SVM =>
//        training.map{ point =>
//          val p = model.predict(point.features)
//          val y = 2.0 * point.label - 1.0
//          assert(y != 0)
//          if (y * p <= 0.0) 1.0 else 0.0
//        }.reduce(_ + _) / numTraining.toDouble
//    }

//    val prediction = model.predict(training.map(_.features))
//    val predictionAndLabel = prediction.zip(training.map(_.label))
//    val metrics = new BinaryClassificationMetrics(predictionAndLabel)


//    val trainingLoss = model.loss(training) * numTraining.toDouble
//    val scaledReg = params.regParam / 2.0
//    val regularizationPenalty = scaledReg * math.pow(model.weights.l2Norm, 2)
//    val totalLoss = trainingLoss + regularizationPenalty

    // unscale the params to be saved in original form
    // if (params.scaled) {
    //   params.lagrangianRho /= numTraining.toDouble // / params.numPartitions.toDouble
    //   params.rho0 /= numTraining.toDouble // / params.numPartitions.toDouble
    //   params.regParam /= numTraining.toDouble
    // }


    val resultsMap = Map(
      "algorithm" -> ("\"" + params.algorithm.toString + "\""),
      "nExamples" -> numTraining,
      "nDim" -> model.weights.size,
      "runtimeMS" -> model.runtimeMS,
      "iterations" -> optimizerStats("iterations"),
      "objective" -> objective,
      "propError" -> propError,
      "loss" -> loss,
      "reg" -> regPenalty,
      "minW" -> model.weights.toArray.min,
      "maxW" -> model.weights.toArray.max,
      "params" -> params.toString,
      "dim" -> nDim.toString,
      "stats" -> mapToJson(optimizerStats)
  
  )


    println("RESULT: " + mapToJson(resultsMap))

    // val summary =
    //   s"RESULT: ${params.algorithm}\t" + 
    //   s"${stats("iterations")}\t" +
    //   s"${params.runtimeMS}\t" +
    //   s"${stats("runtime")}\t" +
    //   s"${metrics.areaUnderROC()}\t" +
    //   s"$trainingLoss\t" +
    //   s"$regularizationPenalty\t" +
    //   s"${trainingLoss + regularizationPenalty}\t" +
    //   s"(min(w) = ${model.weights.toArray.min}, max(w) = ${model.weights.toArray.max} )\t" +
    //   s"${params.toString}\t" +
    //   "{" + stats.map { case (k,v) => s"$k: $v" }.mkString(", ") + "}\t" +
    //   s"${metrics.areaUnderPR()}\t" +
    //   s"${trainingError}" 

    // println(summary)


    sc.stop()
  }



}




//
//val nDim = training.take(1).head.features.size
//
//(params.algorithm, params.objective) match {
//case (GD, Logistic) =>
//val algorithm = new LogisticRegressionWithSGD()
//algorithm.optimizer
//.setNumIterations(1000000)
//.setRuntime(params.runtimeMS)
//.setStepSize(params.eta_0)
//.setUpdater(updater)
//.setRegParam(params.regParam)
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.getLastIterations().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString
//)
//(model, results)
//
//case (GD, SVM) =>
//val algorithm = new SVMWithSGD()
//algorithm.optimizer
//.setNumIterations(1000000)
//.setRuntime(params.runtimeMS)
//.setStepSize(params.eta_0)
//.setUpdater(updater)
//.setRegParam(params.regParam)
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.getLastIterations().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString
//)
//(model, results)
//
//case (ADMM, Logistic) | (MiniBatchADMM, Logistic) =>
//val algorithm = new LRWithADMM(params)
//algorithm.optimizer.regularizationFunction = regularizer
//val startTime = System.nanoTime()
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.iteration.toString,
//"avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"primalAvgNorm" -> norm(algorithm.optimizer.stats.primalAvg(), 2).toString,
//"dualAvgNorm" -> norm(algorithm.optimizer.stats.dualAvg(), 2).toString,
//"consensusNorm" -> model.weights.l2Norm.toString,
//"dualUpdates" -> algorithm.optimizer.stats.avgDualUpdates.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results )
//
//case (ADMM, SVM) | (MiniBatchADMM, SVM) =>
//val algorithm = new SVMWithADMM(params)
//algorithm.optimizer.regularizationFunction = regularizer
//val startTime = System.nanoTime()
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.iteration.toString,
//"avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"primalAvgNorm" -> norm(algorithm.optimizer.stats.primalAvg(), 2).toString,
//"dualAvgNorm" -> norm(algorithm.optimizer.stats.dualAvg(), 2).toString,
//"consensusNorm" -> model.weights.l2Norm.toString,
//"dualUpdates" -> algorithm.optimizer.stats.avgDualUpdates.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results)
//
//case (AsyncADMM, Logistic) | (PORKCHOP, Logistic) =>
//val algorithm = new LRWithAsyncADMM(params)
//algorithm.optimizer.regularizer = regularizer
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.stats.avgLocalIters().x.toString,
//"iterInterval" -> algorithm.optimizer.stats.avgLocalIters().toString,
//"avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
//"avgMsgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
//"avgMsgsRcvd" -> algorithm.optimizer.stats.avgMsgsRcvd().toString,
//"primalAvgNorm" -> norm(algorithm.optimizer.stats.primalAvg(), 2).toString,
//"dualAvgNorm" -> norm(algorithm.optimizer.stats.dualAvg(), 2).toString,
//"consensusNorm" -> model.weights.l2Norm.toString,
//"dualUpdates" -> algorithm.optimizer.stats.avgDualUpdates.toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results)
//
//case (AsyncADMM, SVM) | (PORKCHOP, SVM) =>
//val algorithm = new SVMWithAsyncADMM(params)
//algorithm.optimizer.regularizer = regularizer
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.stats.avgLocalIters().x.toString,
//"iterInterval" -> algorithm.optimizer.stats.avgLocalIters().toString,
//"avgSGDIters" -> algorithm.optimizer.stats.avgSGDIters().toString,
//"avgMsgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
//"avgMsgsRcvd" -> algorithm.optimizer.stats.avgMsgsRcvd().toString,
//"primalAvgNorm" -> norm(algorithm.optimizer.stats.primalAvg(), 2).toString,
//"dualAvgNorm" -> norm(algorithm.optimizer.stats.dualAvg(), 2).toString,
//"consensusNorm" -> model.weights.l2Norm.toString,
//"dualUpdates" -> algorithm.optimizer.stats.avgDualUpdates.toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results)
//case (HOGWILD, Logistic) =>
//val algorithm = new LRWithHOGWILD(params)
//algorithm.optimizer.regularizer = regularizer
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.stats.avgLocalIters().x.toString,
//"avgMsgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
//"avgMsgsRcvd" -> algorithm.optimizer.stats.avgMsgsRcvd().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results)
//case (HOGWILD, SVM) =>
//val algorithm = new SVMWithHOGWILD(params)
//algorithm.optimizer.regularizer = regularizer
//val model = algorithm.run(training).clearThreshold()
//val results =
//Map(
//"iterations" -> algorithm.optimizer.stats.avgLocalIters().x.toString,
//"avgMsgsSent" -> algorithm.optimizer.stats.avgMsgsSent().toString,
//"avgMsgsRcvd" -> algorithm.optimizer.stats.avgMsgsRcvd().toString,
//"runtime" -> algorithm.optimizer.totalTimeMs.toString,
//"stats" -> algorithm.optimizer.stats.toString
//)
//(model, results)
//}
