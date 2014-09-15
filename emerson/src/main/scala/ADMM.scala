package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.optimization.{Optimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


class SGDLocalOptimizer(val subProblemId: Int,
                        val nSubProblems: Int,
                        val nData: Int,
                        val data: Array[(Double, BV[Double])],
                        val lossFun: LossFunction,
                        val params: ADMMParams) extends Serializable with Logging {

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
    WorkerStats(primalVar, dualVar, msgsSent = 0,
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

  var t = 0
  def sgd(endByMS: Long = Long.MaxValue) {
    assert(miniBatchSize <= data.size)
    val lossScaleTerm = data.size.toDouble / (nData.toDouble * miniBatchSize.toDouble)

    var currentTime = System.currentTimeMillis()
    residual = Double.MaxValue
    t = 0
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
      val eta_t = params.eta_0 / math.sqrt(t + 1.0)

      // Take a negative gradient step with step size eta
      axpy(-eta_t, grad, primalVar)

      // Compute residual
      residual = eta_t * norm(grad, 2)

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



class ADMM(val params: ADMMParams, var gradient: LossFunction, 
	   var regularizer: Regularizer) 
  extends Optimizer with Serializable with Logging {

  var iteration = 0
  var solvers: RDD[SGDLocalOptimizer] = null
  var stats: WorkerStats = null
  var totalTimeMs: Long = -1

  def setup(rawData: RDD[(Double, Vector)], initialWeights: Vector) {
    val primal0 = initialWeights.toBreeze
    val nSubProblems = rawData.partitions.length
    val nData = rawData.count
    solvers =
      rawData.mapPartitionsWithIndex { (ind, iter) =>
        val data: Array[(Double, BV[Double])] = iter.map {
          case (label, features) => (label, features.toBreeze)
        }.toArray
        val solver = new SGDLocalOptimizer(ind, nSubProblems = nSubProblems,
          nData = nData.toInt, data, gradient, params)
        // Initialize the primal variable and primal regularizer
        solver.primalVar = primal0.copy
        solver.primalConsensus = primal0.copy
        Iterator(solver)
      }.cache()
      solvers.count

    // rawData.unpersist(true)
    solvers.foreach( f => System.gc() )
  }


  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(rawData: RDD[(Double, Vector)],
    initialWeights: Vector): Vector = {

    setup(rawData, initialWeights)

    println(params)

    val nSolvers = solvers.partitions.length
    val nDim = initialWeights.size

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue

    var primalConsensus = initialWeights.toBreeze.copy

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
      primalConsensus = regularizer.consensus(stats.primalAvg, stats.dualAvg, 
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

    Vectors.fromBreeze(primalConsensus)

  }

}

