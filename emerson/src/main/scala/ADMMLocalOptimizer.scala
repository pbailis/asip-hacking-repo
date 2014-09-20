package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.optimization.{Optimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


class ADMMLocalOptimizer(val subProblemId: Int,
                        val nSubProblems: Int,
                        val nData: Int,
                        val data: Array[(Double, BV[Double])],
                        val lossFun: LossFunction,
                        val params: EmersonParams) extends Serializable with Logging {

  val nDim = data(0)._2.size
  val rnd = new java.util.Random(subProblemId)

  val miniBatchSize = math.min(params.miniBatchSize, data.size)
  val regParamScaled = params.regParam

  @volatile var primalConsensus = BV.zeros[Double](nDim)

  @volatile var primalVar = BV.zeros[Double](nDim)

  @volatile var dualVar = BV.zeros[Double](nDim)

  @volatile var grad = BV.zeros[Double](nDim)

  @volatile var sgdIters = 0

  @volatile var dualIters = 0

  @volatile var residual: Double = Double.MaxValue

  @volatile var rho = params.rho0

  @volatile var localIters = 0

  // Current index into the data
  @volatile var dataInd = 0

  def getStats() = {
    Stats(primalVar, dualVar, msgsSent = 0,
      sgdIters = sgdIters,
      dualUpdates = dualIters,
      dataSize = data.length,
      residual = residual)
  }

  def dualUpdate(rate: Double) {
    // Do the dual update
    dualVar += (primalVar - primalConsensus) * rate
    dualIters += 1
  }

  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    val endByMS = System.currentTimeMillis() + remainingTimeMS
    sgd(endByMS)
  }

  // var t = 0
  def sgd(endByMS: Long = Long.MaxValue) {
    assert(miniBatchSize <= data.size)
    // The lossScaleTerm is composed to several parts:
    //                   Scale up sample to machine  *  Loss Scale Term
    // lossScaleTerm = (dataOnMachine  * (1/miniBatch))  *  (1/N)
    //
    // The Loss scale term comes from that fact that the total loss has
    // the form:
    //
    //    Regularizer              Loss Term
    //  lambda * reg(w)  +  1/N \sum_{i=1}^N f(x_i, w)
    //
    val lossScaleTerm = data.size.toDouble / (nData.toDouble * miniBatchSize.toDouble)

    var currentTime = System.currentTimeMillis()
    residual = Double.MaxValue
    var t = 0
    while(t < params.maxWorkerIterations &&
      residual > params.workerTol &&
      currentTime < endByMS) {
      grad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize == data.length) b else rnd.nextInt(data.length)
        lossFun.addGradient(primalVar, data(ind)._2, data(ind)._1, grad)
        b += 1
      }
      // Normalize the gradient to the batch size
      grad *= lossScaleTerm

      // // Assume loss is of the form  lambda/2 |reg|^2 + 1/n sum_i loss_i
      // val scaledRegParam = params.regParam // / nData.toDouble
      // grad += primalVar * scaledRegParam

      // Add the lagrangian
      grad += dualVar

      // Add the augmenting term
      axpy(rho, primalVar - primalConsensus, grad)

      // Set the learning rate
      val eta_t = params.eta_0 / Math.sqrt(t + 1.0)

      // Take a negative gradient step with step size eta
      axpy(-eta_t, grad, primalVar)

      // Compute residual
      residual = eta_t * norm(grad, 2)

      // Increment iteration counter
      t += 1

      // more coarse grained timeing
      if (t % 100 == 0) {
        currentTime = System.currentTimeMillis()
      }
    }
    println(s"$t \t $residual")

    // Save the last num
    sgdIters = t
  }
}
