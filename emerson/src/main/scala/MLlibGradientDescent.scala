package edu.berkeley.emerson

import java.util.concurrent.TimeUnit

import breeze.linalg.{norm, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.Logging
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


class MLlibGradientDescent extends BasicEmersonOptimizer with Serializable with Logging {

  var iteration = 0
  var stats: Stats = null
  var totalTimeMs: Long = -1

  var data2: RDD[LabeledPoint] = null


  override def initialize(params: EmersonParams,
                 lossFunction: LossFunction, regularizationFunction: Regularizer,
                 initialWeights: BV[Double], data: RDD[Array[(Double, BV[Double])]]) {
    println(params)

    this.data = data
    this.params = params
    this.lossFunction = lossFunction
    this.regularizationFunction = regularizationFunction
    this.initialWeights = initialWeights
    this.data2 = data.flatMap(data => data.iterator.map { case (y, x) => LabeledPoint(y, Vectors.fromBreeze(x)) }).cache()
    this.data2.foreach( f => () )

    nDim = initialWeights.size
    nSubProblems = data.partitions.length

    data.cache()
    val perNodeData = data.map( a => a.length ).collect
    nData = perNodeData.sum


    println(s"Per node data size: ${perNodeData.mkString(",")}")

  }


  def statsMap(): Map[String, String] = {
    Map(
      "iterations" -> iteration.toString,
      "iterInterval" -> stats.avgLocalIters().toString,
      "avgSGDIters" -> stats.avgSGDIters().toString,
      "avgMsgsSent" -> stats.avgMsgsSent().toString,
      "avgMsgsRcvd" -> stats.avgMsgsRcvd().toString,
      "primalAvgNorm" -> norm(stats.primalAvg(), 2).toString,
      "dualAvgNorm" -> norm(stats.dualAvg(), 2).toString,
      "consensusNorm" -> norm(weights, 2).toString,
      "dualUpdates" -> stats.avgDualUpdates.toString,
      "runtime" -> totalTimeMs.toString,
      "stats" -> stats.toString
    )
  }

  var weights: BV[Double] = null

  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(): BV[Double] = {
    weights = initialWeights.copy

    val starttime = System.currentTimeMillis()
    val startTimeNs = System.nanoTime()
    data2.cache()


    val updater = regularizationFunction match {
      case l: L1Regularizer  => new L1Updater()
      case l: L2Regularizer => new SquaredL2Updater()
    }

    val model = lossFunction match {
      case l: LogisticLoss =>
        val algorithm = new LogisticRegressionWithSGD()
        algorithm.optimizer
          .setNumIterations(1000000)
          .setRuntime(params.runtimeMS)
          .setStepSize(params.eta_0)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        val res = algorithm.run(data2, initialWeights = Vectors.fromBreeze(initialWeights.copy)).clearThreshold()
	iteration = algorithm.optimizer.getLastIterations()
	res
      case l: HingeLoss =>
        val algorithm = new SVMWithSGD()
        algorithm.optimizer
          .setNumIterations(1000000)
          .setRuntime(params.runtimeMS)
          .setStepSize(params.eta_0)
          .setUpdater(updater)
          .setRegParam(params.regParam)
        val res = algorithm.run(data2, initialWeights = Vectors.fromBreeze(initialWeights.copy)).clearThreshold()
	iteration = algorithm.optimizer.getLastIterations()
	res
    }

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")

    weights = model.weights.toBreeze

    stats = Stats(weights, BV.zeros[Double](weights.size))

    weights
  }

}

