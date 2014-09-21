package edu.berkeley.emerson


import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging



class DualDecompLocalOptimizer(val subProblemId: Int,
                        val nSubProblems: Int,
                        val nData: Int,
                        val data: Array[(Double, BV[Double])],
                        val lossFunction: LossFunction,
                        val regularizer: Regularizer,
                        val params: EmersonParams,
                        val primal0: BV[Double]) extends Serializable with Logging {

  val nDim = data(0)._2.size
  val rnd = new java.util.Random(subProblemId)

  val miniBatchSize = math.min(params.miniBatchSize, data.size)

  val primalVars = Array.fill(nSubProblems)(primal0.copy)
  val dualVars = Array.fill(nSubProblems)(BV.zeros[Double](nDim))
  def localPrimalVar = primalVars(subProblemId)

  val primalGrad = BV.zeros[Double](nDim)
  val dualSum = BV.zeros[Double](nDim)


  @volatile var sgdIters = 0

  @volatile var dualIters = 0

  @volatile var residual: Double = Double.MaxValue


  @volatile var localIters = 0

  // Current index into the data
  @volatile var dataInd = 0

  def getStats() = {
    dualSum *= 0.0
    var j = 0
    while (j < nSubProblems) {
      dualSum += dualVars(j)
      j += 1
    }
    dualSum /= nSubProblems.toDouble
    Stats(localPrimalVar,  dualSum, msgsSent = 0,
      sgdIters = sgdIters,
      dualUpdates = dualIters,
      dataSize = data.length,
      residual = residual)
  }

  def dualUpdate(rate: Double) {
    assert(dualVars.length == nSubProblems)
    // Do the dual update
    var j = 0
    while (j < nSubProblems) {
      axpy(rate, localPrimalVar - primalVars(j), dualVars(j))
      j += 1
    }
    dualIters += 1
  }


  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    val endByMS = System.currentTimeMillis() + remainingTimeMS
    sgd(endByMS)
  }

  /**
   * Run SGD on the primal
   * @param endByMS
   */
  def sgd(endByMS: Long = Long.MaxValue) {
    assert(miniBatchSize <= data.size)
    // The lossScaleTerm is composed to several parts:
    //                   Scale up sample to machine           Loss Scale Term
    // lossScaleTerm = (dataOnMachine  * (1/miniBatch))   *        (1/N)
    //
    // The Loss scale term comes from that fact that the total loss has
    // the form:
    //
    //    Regularizer              Loss Term
    //  regParam * reg(w)  +  1/N \sum_{i=1}^N f(x_i, w)
    //
    val lossScaleTerm = data.size.toDouble / (nData.toDouble * miniBatchSize.toDouble)
    val scaledRegParam = params.regParam / nSubProblems.toDouble

    // Compute dualSum as defined in:
    dualSum *= 0.0
    var j = 0
    while (j < nSubProblems) {
      dualSum += dualVars(j)
      j += 1
    }

    var currentTime = System.currentTimeMillis()
    residual = Double.MaxValue
    var t = 0
    while(t < params.maxWorkerIterations &&
      residual > params.workerTol &&
      currentTime < endByMS) {
      primalGrad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize == data.length) b else rnd.nextInt(data.length)
        lossFunction.addGradient(localPrimalVar, data(ind)._2, data(ind)._1, primalGrad)
        b += 1
      }

      // Normalize the gradient to the batch size
      primalGrad *= lossScaleTerm

      // Add the dual sum
      primalGrad += dualSum

      // Add the regularizer.
      regularizer.addGradient(localPrimalVar, scaledRegParam, primalGrad)

      // Set the learning rate
      val eta_t = params.eta_0 / math.sqrt(t + 1.0)

      // Scale the gradient by the step size
      primalGrad *= eta_t

      // Subtract the gradient
      localPrimalVar -= primalGrad

      // Compute residual
      residual = norm(primalGrad, 2)

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
