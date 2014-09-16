package edu.berkeley.emerson


import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging

/* Basic Derivation: (paste into latexit)

N & := \text{number of data points} \\
P & := \text{number of machines} \\
n_p & := \text{number of data points on machine $p$} \\
%
%
\mathcal{L}(x, \lambda)
&=
\ underbrace{\left(\frac{1}{N} \sum_{i=1}^P \sum_{j=1}^{n_p} f_{ij}(x_i) \right)}_{\text{loss}} +
\ underbrace{\left(\frac{1}{2P} \sum_{i=1}^P ||x_i||_2^2 \right)}_{\text{regularizer}} +
\ underbrace{\left(\sum_{i=1}^P \sum_{j =i+1}^P \lambda_{ij}^T (x_i - x_j)  \right)
}_{\text{undirected equality constraints}} \\
&=
\sum_{i=1}^P \left(
\frac{1}{N} \sum_{j=1}^{n_p} f_{ij}(x_i) +
\frac{1}{2P} ||x_i||_2^2  +
\sum_{j =i+1}^P \lambda_{ij}^T (x_i - x_j)  \right) \\
%
%
\nabla_{x_i}\mathcal{L}(x, \lambda) & = \frac{1}{N} \sum_{j=1}^{n_p}  \nabla_{x_i}  f_{ij}(x_i)
+ \frac{1}{P} x_i
+ \left(\sum_{j=i+1}^P \lambda_{ij}\right) - \left(\sum_{j=1}^{i-1} \lambda_{ji}\right) \\
%
%
\sigma_i & := \left(\sum_{j=i+1}^P \lambda_{ij}\right) - \left(\sum_{j=1}^{i-1} \lambda_{ji}\right)\\
%
%
\nabla_{x_i}\mathcal{L}(x, \lambda) & = \frac{1}{N} \sum_{j=1}^{n_p}  \nabla_{x_i}  f_{ij}(x_i) +
  \frac{1}{P} x_i  + \sigma_i  \\
%
%
\nabla_{\lambda_{ij}}\mathcal{L}(x, \lambda) & = (x_i - x_j) \quad \quad :i < j \\


Extended Derivation of a condensed version of the \sigma_i update (this optimization is not being used yet)

\lambda^{(t+1)}_{ij} & = \lambda^{(t)}_{ij} + \eta (x_i - x_j) \\
%
%
\sigma^{(t+1)}_i & = \left(\sum_{j=i+1}^P \lambda^{(t)}_{ij} + \eta (x_i - x_j)\right) -
   \left(\sum_{j=1}^{i-1} \lambda^{(t)}_{ji} + \eta (x_j - x_i)\right) \\
& = \left(\sum_{j=i+1}^P \lambda^{(t)}_{ij} - \sum_{j=1}^{i-1} \lambda^{(t)}_{ji} \right) +
   \eta \left(\sum_{j=i+1}^P (x_i - x_j) - \sum_{j=1}^{i-1}  (x_j - x_i) \right) \\
& = \sigma^{(t)}_i + \eta \sum_{j=1}^P (x_i - x_j)


 */

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
  val localPrimalVar = primalVars(subProblemId)

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
    // Do the dual update
    var j = 0
    while (j < nSubProblems) {
      // Upper Triangular Invariant:
      // \lambda_{ij}^T (x_i - x_j) is defined for i < j
      if (j < subProblemId) {
        axpy(rate, primalVars(j) - localPrimalVar, dualVars(j))
      } else {
        axpy(rate, localPrimalVar - primalVars(j), dualVars(j))
      }
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
    // \sigma_i &= \left(\sum_{j=i+1}^P \lambda_{ij}\right) - \left(\sum_{j=1}^{i-1} \lambda_{ji}\right) \\
    dualSum *= 0.0
    var j = 0
    while (j < nSubProblems) {
      if (j < subProblemId) {
        dualSum -= dualVars(j)
      } else if (j > subProblemId) {
        dualSum += dualVars(j)
      }
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
