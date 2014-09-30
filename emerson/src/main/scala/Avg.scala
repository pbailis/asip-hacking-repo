package edu.berkeley.emerson

import java.util.concurrent.TimeUnit

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


class Avg extends BasicEmersonOptimizer with Serializable with Logging {

  var solvers: RDD[ADMMLocalOptimizer] = null
  var stats: Stats = null
  var totalTimeMs: Long = -1

//  def initialize(params: EmersonParams,
//                 lossFunction: LossFunction, regularizationFunction: Regularizer,
//                 initialWeights: BV[Double], rawData: RDD[(Double, BV[Double])]) {
//    println(params)
//
//    this.params = params
//    this.lossFunction = lossFunction
//    this.regularizationFunction = regularizationFunction
//    this.initialWeights = initialWeights
//
//    val primal0 = initialWeights
//    val nSubProblems = rawData.partitions.length
//    val nData = rawData.count
//    solvers =
//      rawData.mapPartitionsWithIndex { (ind, iter) =>
//        val data: Array[(Double, BV[Double])] = iter.toArray
//        val solver = new ADMMLocalOptimizer(ind, nSubProblems = nSubProblems,
//          nData = nData.toInt, data, lossFunction, params)
//        // Initialize the primal variable and primal regularizer
//        solver.primalVar = primal0.copy
//        solver.primalConsensus = primal0.copy
//        Iterator(solver)
//      }.cache()
//      solvers.count
//
//    // rawData.unpersist(true)
//    solvers.foreach( f => System.gc() )
//  }

  def statsMap(): Map[String, String] = {
    Map(
      "iterations" -> stats.avgLocalIters().x.toString,
      "iterInterval" -> stats.avgLocalIters().toString,
      "avgSGDIters" -> stats.avgSGDIters().toString,
      "avgMsgsSent" -> stats.avgMsgsSent().toString,
      "avgMsgsRcvd" -> stats.avgMsgsRcvd().toString,
      "primalAvgNorm" -> norm(stats.primalAvg(), 2).toString,
      "dualAvgNorm" -> norm(stats.dualAvg(), 2).toString,
      "consensusNorm" -> norm(primalConsensus, 2).toString,
      "dualUpdates" -> stats.avgDualUpdates.toString,
      "runtime" -> totalTimeMs.toString,
      "stats" -> stats.toString
    )
  }

  var primalConsensus: BV[Double] = null

  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(): BV[Double] = {
    primalConsensus = initialWeights.copy
    // Force to run forever
    params.maxWorkerIterations = Int.MaxValue

    // Initialize the solvers
    val primal0 = initialWeights
    solvers = data.mapPartitionsWithIndex { (ind, iter) =>
      val data: Array[(Double, BV[Double])] = iter.next()
      val solver = new ADMMLocalOptimizer(ind, nSubProblems = nSubProblems,
        nData = nData.toInt, data, lossFunction, regularizationFunction,  params)
      // Initialize the primal variable and primal regularizer
      solver.primalVar = primal0.copy
      solver.primalConsensus = primal0.copy
      solver.useReg = true // turn on regularization in the primal update
      Iterator(solver)
    }.cache()

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue


    val starttime = System.currentTimeMillis()
    val startTimeNs = System.nanoTime()


    val timeRemaining =
      params.runtimeMS - (System.currentTimeMillis() - starttime)
     
    println(s"Master time remaining ${timeRemaining}")

    // Run the local solvers
    stats = solvers.map { solver =>
      // Make sure that the local solver did not reset!
      assert(solver.localIters == 0)

      // Do a dual update
      solver.primalConsensus = primalConsensus.copy

      // Do a primal update
      solver.primalUpdate(Math.min(timeRemaining, params.localTimeout))

      // Construct stats
      solver.getStats()
    }.reduce( _ + _ )

    // Compute the consensus average
    primalConsensus = stats.primalAvg

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")

    primalConsensus
  }

}

