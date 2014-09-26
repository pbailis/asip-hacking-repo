package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import breeze.optimize.DiffFunction
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
                        val regularizer: Regularizer,
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
    Stats(primalVar.copy, dualVar.copy, msgsSent = 0,
      localIters = localIters,
      sgdIters = sgdIters,
      dualUpdates = dualIters,
      dataSize = data.length,
      residual = residual)
  }

  def dualUpdate(rate: Double) {
    // Do the dual update
    axpy(rate, primalVar - primalConsensus, dualVar)
    dualIters += 1
  }

  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    val endByMS = System.currentTimeMillis() + remainingTimeMS
    // params.useLBFGS = true
    // params.maxWorkerIterations = 10
    // params.workerTol = 1.0e-10

    localIters += 1

    // val preLoss = lossFun(primalVar, data) / nData.toDouble
    // val preDiff = primalVar - primalConsensus
    // val preLag = dualVar.dot(preDiff) + (rho / 2.0) * math.pow(norm(preDiff, 2), 2)
    // val preObj = preLoss + preLag

    if(params.useLBFGS) {
      lbfgs(endByMS)
    } else {
      sgd(endByMS)
    }

    // val postLoss = lossFun(primalVar, data) / nData.toDouble
    // val postDiff = primalVar - primalConsensus
    // val postLag = dualVar.dot(postDiff) + (rho / 2.0) * math.pow(norm(postDiff, 2), 2)
    // val postObj = postLoss + postLag
    // println(s"($preObj, $preLoss, $preLag), ($postObj, $postLoss, $postLag)")

  }

  def lbfgs(remainingTimeMS: Long = Long.MaxValue) {
    val lbfgs = new breeze.optimize.LBFGS[BDV[Double]](//params.maxWorkerIterations,
      tolerance = params.workerTol, m = 5)
    var funEvals = 0
    val f = new DiffFunction[BDV[Double]] {
      override def calculate(x: BDV[Double]) = {
        var obj = 0.0
        var cumGrad = BDV.zeros[Double](x.length)
        obj = lossFun.addGradient(x, data, cumGrad)
        // Scale by size of the data
        cumGrad /= nData.toDouble
        obj /= nData.toDouble

        val diff = x - primalConsensus 

        // Add the dual variable
        cumGrad += dualVar
        obj += dualVar.dot(diff)

        // Add the augmenting term
        axpy(rho, diff, cumGrad)
        obj += (rho / 2.0) *  math.pow(norm(diff, 2), 2)
 
        funEvals += 1
        
        (obj, cumGrad)
      }
    }
    primalVar = lbfgs.minimize(f, primalConsensus.toDenseVector)
    println(s"Function Evals $funEvals")
    sgdIters += funEvals
  }


  def lineSearch(grad: BV[Double], endByMS: Long = Long.MaxValue): Double = {
    var etaBest = 1.0e-10
    var w = primalVar - grad * etaBest
    var diff = w - primalConsensus
    var scoreBest = lossFun(w, data) + dualVar.dot(diff) +
      (rho / 2.0) * math.pow(norm(diff,2), 2)
    var etaProposal = etaBest * 2.0
    w = primalVar - grad * etaProposal
    diff = w - primalConsensus
    var newScoreProposal = lossFun(w, data) + dualVar.dot(diff) +
      (rho / 2.0) * math.pow( norm(diff, 2), 2)
    var searchIters = 0
    // Try to decrease the objective as much as possible
    while (newScoreProposal <= scoreBest) {
      etaBest = etaProposal
      scoreBest = newScoreProposal
      // Double eta and propose again.
      etaProposal *= 2.0
      w = primalVar - grad * etaProposal
      diff = w - primalConsensus
      newScoreProposal = lossFun(w, data) + dualVar.dot(diff)
        (rho / 2.0) * math.pow( norm(diff, 2), 2)
      searchIters += 1
      // // Kill the loop if we run out of search time
      // val currentTime = System.currentTimeMillis()
      // if (currentTime > endByMS) {
      //   // etaProposal = 0.0
      //   newScoreProposal = Double.MaxValue
      //   println(s"Ran out of linesearch time on $searchIters: $currentTime > $endByMS")
      // }
    }
    etaBest
  }



  var learningT = 0.0

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
    val lossScaleTerm = data.length.toDouble / (nData.toDouble * miniBatchSize.toDouble)

    // Reset the primal variable to start at consensus
    // primalVar = primalConsensus.copy

    var currentTime = System.currentTimeMillis()
    residual = Double.MaxValue
    var t = 0

    if (!params.learningT) {
      learningT = 0.0
    }
    while(t < params.maxWorkerIterations &&
      residual > params.workerTol &&
      currentTime < endByMS) {

      // Compute the gradient of the loss
      grad *= 0.0 
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize == data.length) b else rnd.nextInt(data.length)
        lossFun.addGradient(primalVar, data(ind)._2, data(ind)._1, grad)
        b += 1
      }

      // Normalize the gradient to the batch size
      grad *= lossScaleTerm

      // val lossGradNorm = norm(grad, 2)

      // Add the regularization term into the gradient step
      // regularizer.addGradient(primalVar, params.regParam / nSubProblems.toDouble, grad)

      // Add the lagrangian
      grad += dualVar

      // Add the augmenting term
      // grad = grad + rho (primalVar - primalConsensus)
      // axpy(rho, primalVar - primalConsensus, grad)
      // grad = grad + rho * primalVar - rho * primalConsensus
      // grad += rho * primalVar
      // grad -= rho * primalConsensus      
      axpy(rho, primalVar, grad)
      axpy(-rho, primalConsensus, grad)

      // println(s"grad norm: (${norm(primalVar,2)}, $lossGradNorm, ${norm(grad, 2)})")

      // Set the learning rate
      val eta_t = 
        if (params.useLineSearch) {
          lineSearch(grad)
        } else {
          params.eta_0 / Math.sqrt(learningT + 1.0)
        }
      learningT += 1.0

      // Take a negative gradient step with step size eta
      axpy(-eta_t, grad, primalVar)

      // Compute residual and current time in frequently to reduce overhead
      if (t % 100 == 0) {
        residual = eta_t * norm(grad, 2)
        currentTime = System.currentTimeMillis()
      }

      // Increment iteration counter
      t += 1
    }
    println(s"$t \t $residual: \t (primalMin, primalMax): (${min(primalVar.toDenseVector)}, ${max(primalVar.toDenseVector)}")

    // Save the last num
    sgdIters += t
  }
}
