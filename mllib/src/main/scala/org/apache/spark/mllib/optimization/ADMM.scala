package org.apache.spark.mllib.optimization

import java.util.concurrent.TimeUnit

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

trait ObjectiveFunction extends Serializable {
  def addGradient(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double
  def apply(w: BV[Double], x: BV[Double], y: Double): Double = 0.0

  def apply(w: BV[Double], data: Array[(Double, BV[Double])]): Double = {
    var i = 0
    var sum = 0.0
    while (i < data.length) {
      sum += apply(w, data(i)._2, data(i)._1)
      i += 1
    }
    sum 
  }
}






class HingeObjective extends ObjectiveFunction {
  override def addGradient(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double = {
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      axpy(-yscaled, x, cumGrad)
      1.0 - yscaled * wdotx
    } else {
      0.0
    }
  }
  override def apply(w: BV[Double], x: BV[Double], y: Double): Double = {
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      1.0 - yscaled * wdotx
    } else {
      0.0
    }
  }
}

/*
Gradient
P(y \,|\, x,w) &= \left(1 - \sigma(w^T x) \right)^{(1-y)}  \sigma(w^T x)^{y} \\
\log P(y \,|\, x,w) &= (1-y) \log \left(1 - \sigma(w^T x) \right) +  y \log \sigma(w^T x) \\
\nabla_w \log P(y \,|\, x,w) &= \left(-(1-y) \frac{1}{1 - \sigma(w^T x)} +  y  \frac{1}{\sigma(w^T x)}\right) \nabla_w \sigma(w^T x) \\
\nabla_w \log P(y \,|\, x,w) &= \left(-(1-y) \frac{1}{1 - \sigma(w^T x)} +  y  \frac{1}{\sigma(w^T x)}\right) \sigma(w^T x) \left(1-  \sigma(w^T x) \right) \nabla_w (w^t x) \\
\nabla_w \log P(y \,|\, x,w) &= \left(-(1-y) \frac{1}{1 - \sigma(w^T x)} +  y \frac{1}{\sigma(w^T x)}\right) \sigma(w^T x) \left(1-  \sigma(w^T x) \right) x \\
\nabla_w \log P(y \,|\, x,w) &= \left(-(1-y) \frac{\sigma(w^T x) \left(1-  \sigma(w^T x) \right)}{1 - \sigma(w^T x)} +  y \frac{\sigma(w^T x) \left(1-  \sigma(w^T x) \right)}{\sigma(w^T x)}\right)  x \\
\nabla_w \log P(y \,|\, x,w) &= \left(-(1-y) \sigma(w^T x) +  y \left(1-  \sigma(w^T x) \right) \right)  x \\
\nabla_w \log P(y \,|\, x,w) &= \left(-\sigma(w^T x) + y \sigma(w^T x)  +   y -  y \sigma(w^T x)  \right)  x \\
\nabla_w \log P(y \,|\, x,w) &= \left(y -\sigma(w^T x) \right)  x

Likelihood
P(y \,|\, x,w) &= \left(1 - \sigma(w^T x) \right)^{(1-y)}  \sigma(w^T x)^{y} \\
\log P(y \,|\, x,w) &=  y \log \frac{1}{1 + \exp(-w^T x)} + (1-y) \log \left(1 - \frac{1}{1 + \exp(-w^T x)} \right)   \\
\log P(y \,|\, x,w) &=  -y \log \left( 1 + \exp(-w^T x) \right) + (1-y) \log \left(\frac{\exp(-w^T x)}{1 + \exp(-w^T x)} \right) \\
\log P(y \,|\, x,w) &=  -y \log \left( 1 + \exp(-w^T x) \right) + (1-y) \log \exp(-w^T x) - (1-y) \log \left( 1 + \exp(-w^T x) \right)  \\
\log P(y \,|\, x,w) &=  (1 - y) (-w^T x) -\log \left( 1 + \exp(-w^T x) \right)
 */
class LogisticObjective extends ObjectiveFunction {
  def sigmoid(x: Double) = 1.0 / (1.0 + math.exp(-x))
  override def addGradient(w: BV[Double], x: BV[Double], label: Double, cumGrad: BV[Double]) = {
    val wdotx = w.dot(x)
    val gradientMultiplier = label - sigmoid(wdotx)
    // Note we negate the gradient here since we ant to minimize the negative of the likelihood
    axpy(-gradientMultiplier, x, cumGrad)
    val logLikelihood =
      if (label > 0) {
        -math.log1p(math.exp(-wdotx)) // log1p(x) = log(1+x)
      } else {
        -wdotx - math.log1p(math.exp(-wdotx))
      }
    -logLikelihood
  }
  override def apply(w: BV[Double], x: BV[Double], label: Double) = {
    val wdotx = w.dot(x)
    val logLikelihood =
      if (label > 0) {
        -math.log1p(math.exp(-wdotx)) // log1p(x) = log(1+x)
      } else {
        -wdotx - math.log1p(math.exp(-wdotx))
      }
    -logLikelihood
  }
}








trait ConsensusFunction extends Serializable {
  def apply(primalAvg: BV[Double], dualAvg: BV[Double], nSolvers: Int, rho: Double, regParam: Double): BV[Double]
}

/*
0 & = \nabla_z \left( \lambda ||z||_2^2 + \sum_{i=1}^N \left( \mu_i^T (x_i - z) +  \frac{\rho}{2} ||x_i - z||_2^2 \right)  \right) \\
0 & = \lambda z - \sum_{i=1}^N \left(\mu_i + \rho (x_i - z) \right)  \\
0 & =\lambda z - N \bar{u} - \rho N \bar{x} + \rho N z    \\
0 & = z (\lambda + \rho N) -  N (\bar{u} + \rho \bar{x} )  \\
z & = \frac{ N}{\lambda + \rho N} (\bar{u} + \rho \bar{x})
z & = \frac{ \rho N}{\lambda + \rho N} (\frac{1}{\rho}\bar{u} + \bar{x})
*/
class L2ConsensusFunction extends ConsensusFunction {
  override def apply(primalAvg: BV[Double], dualAvg: BV[Double], nSolvers: Int, rho: Double, regParam: Double): BV[Double] = {
    val nDim = dualAvg.size
    assert(nDim > 0)
    val rhoScaled = rho
    val regScaled = 0.0 // regParam
    assert((regScaled + nSolvers * rhoScaled) > 0)
    assert(nDim.toDouble > 0)
    if (rho == 0.0) {
      primalAvg 
    } else {
      val multiplier = (nSolvers.toDouble) / (regScaled + nSolvers * rhoScaled)
      (primalAvg * rhoScaled + dualAvg) * multiplier
    }
  }
}

class L1ConsensusFunction extends ConsensusFunction {
  def softThreshold(alpha: Double, x: BV[Double]): BV[Double] = {
    val ret = BV.zeros[Double](x.size)
    var i = 0
    while (i < x.size) {
      if(x(i) < alpha) {
        ret(i) = x(i) + alpha
      } else if (x(i) > alpha) {
        ret(i) = x(i) - alpha
      }
      i +=1
    }
    ret
  }
  override def apply(primalAvg: BV[Double], dualAvg: BV[Double], nSolvers: Int, rho: Double, regParam: Double): BV[Double] = {
    val nDim = dualAvg.size
    val rhoScaled = rho
    val regScaled = regParam
    if (rho == 0.0) {
      softThreshold(regParam, primalAvg)
    } else {
      val threshold = regScaled / (nSolvers.toDouble * rhoScaled)
      softThreshold(threshold, primalAvg + dualAvg / rhoScaled)
    }
  }
}







object Interval {
  def apply(x: Int) = new Interval(x)
  def apply(x: Double) = new Interval(x)
}



class Interval(val x: Double, val xMin: Double, val xMax: Double) extends Serializable {
  def this(x: Double) = this(x, x, x)
  def +(other: Interval) = {
    new Interval(x+other.x, math.min(xMin, other.xMin), math.max(xMax, other.xMax))
  }
  def /(d: Double) = new Interval(x / d, xMin, xMax)

  override def toString = s"[$xMin, $x, $xMax]"
}



object WorkerStats {
  def apply(primalVar: BV[Double], dualVar: BV[Double],
    msgsSent: Int = 0,
    msgsRcvd: Int = 0,
    localIters: Int = 0,
    sgdIters: Int = 0,
    dualUpdates: Int = 0,
    residual: Double = 0.0,
    dataSize: Int = 0) = {
    new WorkerStats(
      weightedPrimalVar = primalVar,
      weightedDualVar = dualVar,
      msgsSent = Interval(msgsSent),
      msgsRcvd = Interval(msgsRcvd),
      localIters = Interval(localIters),
      sgdIters = Interval(sgdIters),
      dualUpdates = Interval(dualUpdates),
      residual = Interval(residual),
      dataSize = Interval(dataSize),
      nWorkers = 1)
  }
}



case class WorkerStats(
  weightedPrimalVar: BV[Double],
  weightedDualVar: BV[Double],
  msgsSent: Interval,
  msgsRcvd: Interval,
  localIters: Interval,
  sgdIters: Interval,
  dualUpdates: Interval,
  dataSize: Interval,
  residual: Interval,
  nWorkers: Int) extends Serializable {

  def withoutVars() = {
    WorkerStats(null, null, 
      msgsSent = msgsSent, 
      msgsRcvd = msgsRcvd,
      localIters = localIters, 
      sgdIters = sgdIters, 
      dualUpdates = dualUpdates, 
      dataSize = dataSize, 
      residual = residual, 
      nWorkers = nWorkers)
  }

  def +(other: WorkerStats) = {
    new WorkerStats(
      weightedPrimalVar = weightedPrimalVar + other.weightedPrimalVar,
      weightedDualVar = weightedDualVar + other.weightedDualVar,
      msgsSent = msgsSent + other.msgsSent,
      msgsRcvd = msgsRcvd + other.msgsRcvd,
      localIters = localIters + other.localIters,
      sgdIters = sgdIters + other.sgdIters,
      dualUpdates = dualUpdates + other.dualUpdates,
      dataSize = dataSize + other.dataSize,
      residual = residual + other.residual,
      nWorkers = nWorkers + other.nWorkers)
  }

  def toMap(): Map[String, Any] = {
    Map(
      "primalAvg" -> ("[" + primalAvg().toArray.mkString(", ") + "]"),
      "dualAvg" -> ("[" + dualAvg().toArray.mkString(", ") + "]"),
      "avgMsgsSent" -> avgMsgsSent(),
      "avgMsgsRcvd" -> avgMsgsRcvd(),
      "avgLocalIters" -> avgLocalIters(),
      "avgDualUpdates" -> avgDualUpdates(),
      "avgSGDIters" -> avgSGDIters(),
      "avgResidual" -> avgResidual()
    )
  }

  override def toString = {
    "{" + toMap.iterator.map { 
      case (k,v) => "\"" + k + "\": " + v 
    }.toArray.mkString(", ") + "}"
  }

  def primalAvg(): BV[Double] = {
    if (weightedPrimalVar == null) null else weightedPrimalVar / nWorkers.toDouble
  }
  def dualAvg(): BV[Double] = {
    if (weightedDualVar == null) null else weightedDualVar / nWorkers.toDouble
  }
  def avgMsgsSent() = msgsSent / nWorkers.toDouble
  def avgMsgsRcvd() = msgsRcvd / nWorkers.toDouble
  def avgLocalIters() = localIters / nWorkers.toDouble
  def avgSGDIters() = sgdIters / nWorkers.toDouble
  def avgDualUpdates() = dualUpdates / nWorkers.toDouble
  def avgResidual() = residual / nWorkers.toDouble
}









class ADMMParams extends Serializable {
  var eta_0 = 1.0
  var tol = 1.0e-5
  var workerTol = 1.0e-5
  var maxIterations = 1000
  var maxWorkerIterations = 1000
  var miniBatchSize = 10
  var useLBFGS = false
  var rho0 = 1.0
  var lagrangianRho = 1.0
  var regParam = 0.1
  var runtimeMS = Int.MaxValue
  var displayIncrementalStats = false
  var adaptiveRho = false
  var broadcastDelayMS = 100
  var usePorkChop = false
  var useLineSearch = false
  var localTimeout = Int.MaxValue

  def toMap(): Map[String, Any] = {
    Map(
      "eta0" -> eta_0,
      "tol" -> tol,
      "workerTol" -> workerTol,
      "maxIterations" -> maxIterations, 
      "maxWorkerIterations"  -> maxWorkerIterations,
      "miniBatchSize" -> miniBatchSize,
      "useLBFGS" -> useLBFGS,
      "rho0" -> rho0,
      "lagrangianRho" -> lagrangianRho,
      "regParam" -> regParam,
      "runtimeMS" -> runtimeMS,
      "displayIncrementalStats" -> displayIncrementalStats,
      "adaptiveRho" -> adaptiveRho,
      "useLineSearch" -> useLineSearch,
      "broadcastDelayMS" -> broadcastDelayMS,
      "usePorkChop" -> usePorkChop,
      "localTimeout" -> localTimeout
    )
  }
  override def toString = {
    "{" + toMap.iterator.map { 
      case (k,v) => "\"" + k + "\": " + v 
    }.toArray.mkString(", ") + "}"
  }
}



@DeveloperApi
class SGDLocalOptimizer(val subProblemId: Int,
                        val nSubProblems: Int,
                        val nData: Int,
                        val data: Array[(Double, BV[Double])],
                        val objFun: ObjectiveFunction,
                        val params: ADMMParams) extends Serializable with Logging {

  val nDim = data(0)._2.size
  val rnd = new java.util.Random(subProblemId)

  val miniBatchSize = math.min(params.miniBatchSize, data.size)

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
    dualVar = dualVar + (primalVar - primalConsensus) * rate
    dualIters += 1
  }

  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    val endByMS = System.currentTimeMillis() + remainingTimeMS
    sgd(endByMS)
  }

  var t = 0
  def sgd(endByMS: Long = Long.MaxValue) {
    assert(miniBatchSize <= data.size)
    var timeOut = false
    val rhoScaled = rho 
    residual = Double.MaxValue
    t = 0
    while(t < params.maxWorkerIterations && 
      residual > params.workerTol &&
      !timeOut) {
      grad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize == data.length) b else rnd.nextInt(data.length)
        objFun.addGradient(primalVar, data(ind)._2, data(ind)._1, grad)
        b += 1
      }
      // Normalize the gradient to the batch size
      grad /= miniBatchSize.toDouble
      // Assume loss is of the form  lambda/2 |reg|^2 + 1/n sum_i loss_i
      val scaledRegParam = params.regParam // / nData.toDouble
      grad += (primalVar * scaledRegParam)

      // Add the lagrangian
      grad += dualVar
      // Add the augmenting term
      axpy(rhoScaled, primalVar - primalConsensus, grad)  // SCALED TERM
      // Set the learning rate
      // val eta_t = params.eta_0 / ( nDim.toDouble * (t + 1.0) * norm(grad, 2) )
      // math.pow(t + 1.0, 2.0 / 3.0))
      val eta_t = params.eta_0 / (t + 1.0).toDouble
      // Do the gradient update
      primalVar = primalVar - (grad * eta_t)
      // axpy(-eta_t, grad, primalVar)
      // Compute residual.
      residual = eta_t * norm(grad, 2)
      // println(residual)
      t += 1
      timeOut = System.currentTimeMillis() > endByMS
    }
    // Save the last num
    sgdIters = t
  }
}



class ADMM(val params: ADMMParams, var gradient: ObjectiveFunction, var consensus: ConsensusFunction) extends Optimizer with Serializable with Logging {

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
        // Initialize the primal variable and primal consensus
        solver.primalVar = primal0.copy
        solver.primalConsensus = primal0.copy
        Iterator(solver)
      }.cache()
      solvers.count

  //   rawData.unpersist(true)
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

        if(params.adaptiveRho) {
          solver.dualUpdate(rho)
        } else {
          solver.dualUpdate(solver.params.lagrangianRho)
        }

        // Do a primal update
        solver.primalUpdate(Math.min(timeRemaining, params.localTimeout))

        // Construct stats
        solver.getStats()
      }.reduce( _ + _ )

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      primalConsensus = consensus(stats.primalAvg, stats.dualAvg, stats.nWorkers, rho,
        params.regParam)

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
      // println(stats.withoutVars())
      // //println(stats)
      // //println(primalConsensus)
      // println(s"Iteration: $iteration")
      // println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")
      iteration += 1
    }

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")

    Vectors.fromBreeze(primalConsensus)

  }

}

