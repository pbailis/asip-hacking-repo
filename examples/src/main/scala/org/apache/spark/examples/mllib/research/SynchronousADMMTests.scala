package org.apache.spark.examples.mllib.research

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.admm.PegasosSVM
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMWithADMM, LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SquaredL2Updater, L1Updater}



object DataLoaders {
  def loadBismark(sc: SparkContext, filename: String): RDD[LabeledPoint] = {
    val data = sc.textFile(filename)
      .filter(s => !s.isEmpty && s(0) == '{')
      .map(s => s.split('\t'))
      .map { case Array(x, y) =>
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


  def loadFlights(sc: SparkContext, filename: String): RDD[LabeledPoint] = {
    val labels = Array("Year", "Month", "DayOfMonth", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime",
      "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime",
      "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiIn", "TaxiOut",
      "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay",
      "NASDelay", "SecurityDelay", "LateAircraftDelay").zipWithIndex.toMap
    println("Loading data")
    val rawData = sc.textFile(filename, 128).
      filter(s => !s.contains("Year")).
      map(s => s.split(",")).cache()

    val carrierDict = makeDictionary(labels("UniqueCarrier"), rawData)
    val flightNumDict = makeDictionary(labels("FlightNum"), rawData)
    val tailNumDict = makeDictionary(labels("TailNum"), rawData)
    val originDict = makeDictionary(labels("Origin"), rawData)
    val destDict = makeDictionary(labels("Dest"), rawData)

    val data = rawData.map { row =>
      val firstFiveFeatures = (row.view(0, 5) ++ row.view(6, 7)).map{ x => 
      	  if(x == "NA") 0.0 else x.toDouble
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
      LabeledPoint(label, new DenseVector(features))
    }.cache()
    data.count
    println("FINISHED LOADING SUCKA")
    println(s"THIS MANY PLUSES SUCKA ${data.filter(x => x.label == 1).count/data.count.toDouble}")
    data
  }
}


object SynchronousADMMTests {

  object Algorithm extends Enumeration {
    type Algorithm = Value
    val SVM, LR, SVMADMM, Pegasos, PegasosAsync = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import Algorithm._
  import RegType._

  case class Params(
                     input: String = null,
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     algorithm: Algorithm = LR,
                     regType: RegType = L2,
                     regParam: Double = 0.1,
                     format: String = "libsvm")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("BinaryClassification") {
      head("BinaryClassification: an example app for binary classification.")
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
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
      arg[String]("<input>")
        .required()
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input = x))
      arg[String]("format")
        .text("File format")
        .action((x, c) => c.copy(format = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.BinaryClassification \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --algorithm LR --regType L2 --regParam 1.0 \
          |  data/mllib/sample_binary_classification_data.txt
        """.stripMargin)
    }



    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"BinaryClassification with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val examples = if(params.format == "lisbsvm") {
      MLUtils.loadLibSVMFile(sc, params.input).cache()
    } else if (params.format == "bismarck") {
      DataLoaders.loadBismark(sc, params.input).cache()
    } else if (params.format == "flights") {
      DataLoaders.loadFlights(sc, params.input).cache()
    } else {
      throw new RuntimeException("F off")
    }

    val splits = examples.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()

    println(s"defaultparallelism: ${sc.defaultParallelism} minpart: ${sc.defaultMinPartitions}")

    println(s"Training: $numTraining, test: $numTest.")

    examples.unpersist(blocking = false)

    println("STARTING SUCKA")

    val updater = params.regType match {
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val model = params.algorithm match {
      case LR =>
        val algorithm = new LogisticRegressionWithSGD()
        algorithm.optimizer
          .setNumIterations(params.numIterations)
          .setStepSize(params.stepSize)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        algorithm.run(training).clearThreshold()
      case SVM =>
        val algorithm = new SVMWithSGD()
        algorithm.optimizer
          .setNumIterations(params.numIterations)
          .setStepSize(params.stepSize)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        algorithm.run(training).clearThreshold()
      case SVMADMM =>
        val algorithm = new SVMWithADMM()
        algorithm.maxGlobalIterations = params.numIterations
        algorithm.updater = updater
        algorithm.regParam = params.regParam
        algorithm.run(training).clearThreshold()
      case Pegasos =>
        val algorithm = new PegasosSVM()
        algorithm.run(training)
      case PegasosAsync =>
        val algorithm = new PegasosSVM(async = true)
        algorithm.run(training)

    }

    val prediction = model.predict(test.map(_.features))
    val predictionAndLabel = prediction.zip(test.map(_.label))

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    println(s"Test areaUnderPR = ${metrics.areaUnderPR()}.")
    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")

    sc.stop()
  }

}
