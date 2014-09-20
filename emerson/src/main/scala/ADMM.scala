package edu.berkeley.emerson

import java.util.concurrent.TimeUnit

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


class ADMM extends BasicEmersonOptimizer with Serializable with Logging {

  var iteration = 0
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

    // Initialize the solvers
    val primal0 = initialWeights
    solvers = data.mapPartitionsWithIndex { (ind, iter) =>
      val data: Array[(Double, BV[Double])] = iter.next()
      val solver = new ADMMLocalOptimizer(ind, nSubProblems = nSubProblems,
        nData = nData.toInt, data, lossFunction, params)
      // Initialize the primal variable and primal regularizer
      solver.primalVar = primal0.copy
      solver.primalConsensus = primal0.copy
      Iterator(solver)
    }.cache()

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue


    val starttime = System.currentTimeMillis()
    val startTimeNs = System.nanoTime()

    var rho = params.rho0

    iteration = 0
    while (iteration < params.maxIterations &&
      (primalResidual > params.tol || dualResidual > params.tol) &&
      (System.currentTimeMillis() - starttime) < params.runtimeMS ) {
      println("========================================================")
      println(s"Starting iteration $iteration.")

      val timeRemaining = params.runtimeMS - (System.currentTimeMillis() - starttime)
      // Run the local solvers
      stats = solvers.map { solver =>
        // Make sure that the local solver did not reset!
        assert(solver.localIters == iteration)
        solver.localIters += 1

        // Do a dual update
        solver.primalConsensus = primalConsensus.copy
        solver.primalVar = primalConsensus.copy
        solver.rho = rho

        // if(params.adaptiveRho) {
        //   solver.dualUpdate(rho)
        // } else {
        //   solver.dualUpdate(solver.params.lagrangianRho)
        // }

        solver.dualUpdate(solver.params.lagrangianRho)

        // Do a primal update
        solver.primalUpdate(Math.min(timeRemaining, params.localTimeout))

        // Construct stats
        solver.getStats()
      }.reduce( _ + _ )

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      val regParamScaled = params.regParam // * params.admmRegFactor
      primalConsensus = regularizationFunction.consensus(stats.primalAvg, stats.dualAvg, 
					      stats.nWorkers, rho,
					      regParam = regParamScaled)

      // // Compute the residuals
      // primalResidual = solvers.map(
      //   s => norm(s.primalVar - primalConsensus, 2) * s.data.length.toDouble)
      //   .reduce(_+_) / nSolvers.toDouble
      // dualResidual = norm(primalConsensus - primalConsensusOld, 2)
      // if (params.adaptiveRho) {
      //   if (rho == 0.0) {
      //     rho = 1.0
      //   } else if (primalResidual > 10.0 * dualResidual && rho < 8.0) {
      //     rho = 2.0 * rho
      //     println(s"Increasing rho: $rho")
      //   } else if (dualResidual > 10.0 * primalResidual && rho > 0.1) {
      //     rho = rho / 2.0
      //     println(s"Decreasing rho: $rho")
      //   }
      // }

      println(s"Iteration: $iteration")
      println(stats)
      // println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")
      iteration += 1
    }

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")

    primalConsensus
  }

}

