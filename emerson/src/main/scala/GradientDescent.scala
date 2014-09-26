package edu.berkeley.emerson

import java.util.concurrent.TimeUnit

import breeze.linalg.{norm, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.Logging


class GradientDescent extends BasicEmersonOptimizer with Serializable with Logging {

  var iteration = 0
  var stats: Stats = null
  var totalTimeMs: Long = -1

  def statsMap(): Map[String, String] = {
    Map(
      "iterations" -> iteration.toString,
      "iterInterval" -> stats.avgLocalIters().toString,
      "avgSGDIters" -> stats.avgSGDIters().toString,
      "avgMsgsSent" -> stats.avgMsgsSent().toString,
      "avgMsgsRcvd" -> stats.avgMsgsRcvd().toString,
      "primalAvgNorm" -> norm(weights, 2).toString,
      "dualAvgNorm" -> "0.0",
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
    data.cache()

    iteration = 0
    while (iteration < params.maxIterations &&
      (System.currentTimeMillis() - starttime) < params.runtimeMS ) {
      println("========================================================")
      println(s"Starting iteration $iteration.")
      
      // Compute the loss gradient
      val grad = data.map { data => 
        val grad = BV.zeros[Double](nDim)
        lossFunction.addGradient(weights, data, grad)
        grad
      }.reduce(_ + _) / nData.toDouble

      // Add the regularization penalty
      this.regularizationFunction.addGradient(weights, params.regParam, grad)

      // Set the learning rate
      val eta_t = params.eta_0 / math.sqrt(iteration.toDouble + 1.0)

      // Scale the gradient by the learning rate
      grad *= eta_t

      // Take the gradient step
      weights -= grad

      println(s"Iteration: $iteration")
      // println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")
      iteration += 1
    }


    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    stats = Stats(weights, BV.zeros[Double](weights.size))

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")

    weights
  }

}

