package org.apache.spark.mllib.optimization

import java.util.concurrent.TimeUnit

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import breeze.optimize.DiffFunction
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


trait FastGradient extends Serializable {
  def apply(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double])
  def gradAndValue(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double = 0.0
}

class FastHingeGradient extends FastGradient {
  override def apply(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]) {
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      axpy(-yscaled, x, cumGrad)
    }
  }
  override def gradAndValue(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double = {
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      axpy(-yscaled, x, cumGrad)
      1.0 - yscaled * wdotx
    } else {
      0.0
    }
  }
}


class FastLogisticGradient extends FastGradient {
  override def apply(w: BV[Double], x: BV[Double], label: Double, cumGrad: BV[Double]) {
    val wdotx = w.dot(x)
    val margin = -1.0 * wdotx
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
    axpy(gradientMultiplier, x, cumGrad)
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
*/

class L2ConsensusFunction extends ConsensusFunction {
  override def apply(primalAvg: BV[Double], dualAvg: BV[Double], nSolvers: Int, rho: Double, regParam: Double): BV[Double] = {
    (primalAvg * rho + dualAvg) * (nSolvers.toDouble / (regParam + nSolvers * rho))
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
    assert(false)
    if (rho == 0.0) {
      softThreshold(regParam, primalAvg)
    } else {
      // Joey: rederive this equation:
      softThreshold(regParam / nSolvers.toDouble, primalAvg * rho + dualAvg)
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

  override def toString() = s"[$xMin, $x, $xMax]"
}

object WorkerStats {
  def apply(primalVar: BV[Double], dualVar: BV[Double],
    msgsSent: Int = 0, 
    localIters: Int = 0,
    sgdIters: Int = 0,
    residual: Double = 0.0,
    dataSize: Int = 0) = {
    new WorkerStats(
      weightedPrimalVar = primalVar * dataSize.toDouble, 
      weightedDualVar = dualVar * dataSize.toDouble,
      msgsSent = Interval(msgsSent), 
      localIters = Interval(localIters), 
      sgdIters = Interval(sgdIters),
      residual = Interval(residual),
      dataSize = Interval(dataSize), 
      nWorkers = 1)
  }
}

case class WorkerStats(
  val weightedPrimalVar: BV[Double],
  val weightedDualVar: BV[Double],
  val msgsSent: Interval,
  val localIters: Interval,
  val sgdIters: Interval,
  val dataSize: Interval,
  val residual: Interval,
  val nWorkers: Int) extends Serializable {
  

  def +(other: WorkerStats) = {
    new WorkerStats(
      weightedPrimalVar = weightedPrimalVar + other.weightedPrimalVar,
      weightedDualVar = weightedDualVar + other.weightedDualVar,
      msgsSent = msgsSent + other.msgsSent,
      localIters = localIters + other.localIters, 
      sgdIters = sgdIters + other.sgdIters,
      dataSize = dataSize + other.dataSize,
      residual = residual + other.residual,
      nWorkers = nWorkers + other.nWorkers)
  }

  override def toString() = {
    s"{primalAvg: ${primalAvg()}, dualAvg: ${dualAvg()}, avgMsgsSent: ${avgMsgsSent()}, " +
    s"avgLocalIters: ${avgLocalIters()}, avgSGDIters: ${avgSGDIters()}, avgResidual: ${avgResidual()}}" 
  }

  def primalAvg() = weightedPrimalVar / dataSize.x
  def dualAvg() = weightedDualVar / dataSize.x
  def avgMsgsSent() = msgsSent / nWorkers.toDouble
  def avgLocalIters() = localIters / nWorkers.toDouble
  def avgSGDIters() = sgdIters / nWorkers.toDouble
  def avgResidual() = residual / nWorkers.toDouble

}


class ADMMParams {
  var eta_0 = 1.0
  var tol = 1.0e-5
  var workerTol = 1.0e-5
  var maxIterations = 1000
  var maxWorkerIterations = 1000
  var miniBatchSize = 10
  var useLBFGS = false
}


@DeveloperApi
class SGDLocalOptimizer(val subProblemId: Int,
                        val data: Array[(Double, BV[Double])],
                        val gradient: FastGradient,
                        val params: ADMMParams) extends Serializable with Logging {

  val nExamples = data.length
  val dim = data(0)._2.size
  val rnd = new java.util.Random(subProblemId)  

  @volatile var primalConsensus = BV.zeros[Double](dim)

  @volatile var primalVar = BV.zeros[Double](dim)

  @volatile var dualVar = BV.zeros[Double](dim)
  
  @volatile var grad = BV.zeros[Double](dim)
  

  @volatile var sgdIters = 0

  @volatile var residual: Double = Double.MaxValue

  @volatile var rho = 1.0

  @volatile var localIters = 0

  def getStats() = {
    WorkerStats(primalVar, dualVar, msgsSent = 0, sgdIters = sgdIters, dataSize = data.length, residual = residual)  
  }

  def dualUpdate(rate: Double) {
    // Do the dual update
    dualVar += (primalVar - primalConsensus) * rate
  }

  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    if(params.useLBFGS) {
      lbfgs(remainingTimeMS)
    } else {
      sgd(remainingTimeMS)
    }
  }

  def lbfgs(remainingTimeMS: Long = Long.MaxValue) {
    val lbfgs = new breeze.optimize.LBFGS[BDV[Double]](maxIterations)
    val f = new DiffFunction[BDV[Double]] {
      override def calculate(x: BDV[Double]) = {
        var obj = 0.0
        var cumGrad = BDV.zeros[Double](x.length)
        var i = 0
        while (i < data.length) {
          obj += gradient.gradAndValue(x, data(i)._2, data(i)._1, cumGrad)
          i += 1
        }
        cumGrad /= data.length.toDouble
        obj /= data.length.toDouble

        cumGrad += dualVar
        obj += dualVar.dot(x - primalConsensus)

        axpy(rho, x - primalConsensus, cumGrad)
        obj += (rho / 2.0) *  math.pow(norm(x - primalConsensus, 2), 2)

        (obj, cumGrad)
      }
    }
    primalVar = lbfgs.minimize(f, primalConsensus.toDenseVector)
  }


  def sgd(remainingTimeMS: Long = Long.MaxValue) {
    residual = Double.MaxValue
    val startTime = System.currentTimeMillis()
    var currentTime = startTime
    var t = 0
    while(t < maxIterations && residual > epsilon && (currentTime - startTime) < remainingTimeMS) {
      grad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize < nExamples) rnd.nextInt(nExamples) else b
        gradient(primalVar, data(ind)._2, data(ind)._1, grad)
        b += 1
      }
      // Normalize the gradient to the batch size
      grad /= miniBatchSize.toDouble
      // Add the lagrangian
      grad += dualVar
      // Add the augmenting term
      axpy(rho, primalVar - primalConsensus, grad)
      // Set the learning rate
      val eta_t = eta_0 / (t.toDouble + 1.0)
      // Do the gradient update
      axpy(-eta_t, grad, primalVar)
      // Compute residual.
      residual = eta_t * norm(grad, 2.0)
      // Update the current time every 1000 iterations
      if (t % 100 == 0) {
        currentTime = System.currentTimeMillis()
      }
      t += 1
    }
    // Save the last num
    sgdIters = t
  }
}



class ADMM(var gradient: FastGradient, var consensus: ConsensusFunction) extends Optimizer with Serializable with Logging {

  var numIterations: Int = 100
  var regParam: Double = 1.0
  var epsilon: Double = 1.0e-5
  var eta_0: Double = 1.0
  var localEpsilon: Double = 0.001
  var localMaxIterations: Int = Integer.MAX_VALUE
  var miniBatchSize: Int = 10  // math.max(1, math.min(nExamples, (miniBatchFraction * nExamples).toInt))
  var displayLocalStats: Boolean = true
  var runtimeMS: Int =  Integer.MAX_VALUE
  var rho: Double = 1.0

  var iteration = 0

  var solvers: RDD[SGDLocalOptimizer] = null

  var stats: WorkerStats = null

  def setup(rawData: RDD[(Double, Vector)], initialWeights: Vector) {
    val primal0 = initialWeights.toBreeze
    solvers =
      rawData.mapPartitionsWithIndex { (ind, iter) =>
        val data: Array[(Double, BV[Double])] = iter.map { 
          case (label, features) => (label, features.toBreeze)
        }.toArray
        val solver = new SGDLocalOptimizer(ind, data, gradient)
        solver.eta_0 = eta_0
        solver.epsilon = localEpsilon
        solver.maxIterations = localMaxIterations
        solver.miniBatchSize = miniBatchSize

        // Initialize the primal variable and primal consensus
        solver.primalVar = primal0.copy
        solver.primalConsensus = primal0.copy
        Iterator(solver)
      }.cache()
      solvers.count

    rawData.unpersist(true)
    solvers.foreach( f => System.gc() )
  }

  var totalTimeMs: Long = -1
  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(rawData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    setup(rawData, initialWeights)


    val nDim = initialWeights.size
    val nSolvers = solvers.partitions.length


    //    println(s"nExamples: $nExamples")
    println(s"dim: $nDim")
    println(s"number of solver $nSolvers")

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue

    var primalConsensus = initialWeights.toBreeze.copy

    val starttime = System.currentTimeMillis()

    val startTimeNs = System.nanoTime()

    iteration = 0
    println(s"ADMM numIterations: $numIterations")
    while (iteration < numIterations && (primalResidual > epsilon || dualResidual > epsilon) &&
      (System.currentTimeMillis() - starttime) < runtimeMS ) {
      println("========================================================")
      println(s"Starting iteration $iteration.")
      val timeRemaining = runtimeMS - (System.currentTimeMillis() - starttime)
     
      // Run the local solvers
      stats = solvers.map{ solver =>
        // Make sure that the local solver did not reset!
        assert(solver.localIters == iteration)
        solver.localIters += 1

        // Do a dual update
        solver.primalConsensus = primalConsensus.copy
        solver.rho = rho
        solver.dualUpdate(rho)

        // Do a primal update
        solver.primalUpdate(timeRemaining)

        // Construct stats
        solver.getStats()
        }.reduce( _ + _ )

      println(stats)

      // solvers.map(s => (s.primalVar, s.dualVar)).collect().foreach(x => println(s"\t ${x._1} \t ${x._2}"))

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      primalConsensus = consensus(stats.primalAvg, stats.dualAvg, stats.nWorkers, rho, regParam)

      // println(s"PrimalAvg: ${stats.primalAvg}")
      // println(s"DualAvg: ${stats.dualAvg}")
      // println(s"Consenus: $primalConsensus")

      // // // Compute the residuals
      primalResidual = solvers.map( s => norm(s.primalVar - primalConsensus, 2) * s.data.length)
        .reduce(_+_) / stats.dataSize.x
      dualResidual = rho * norm(primalConsensus - primalConsensusOld, 2)

     // Rho upate from Boyd text
//     if (rho == 0.0) {
//       rho = 1.0
//     } else if (primalResidual > 10.0 * dualResidual && rho < 8.0) {
//       rho = 2.0 * rho
//       println(s"Increasing rho: $rho")
//     } else if (dualResidual > 10.0 * primalResidual && rho > 0.1) {
//       rho = rho / 2.0
//       println(s"Decreasing rho: $rho")
//     }

      println(s"Iteration: $iteration")
      println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")

      iteration += 1
    }
    primalConsensus = consensus(stats.primalAvg, stats.dualAvg, stats.nWorkers, rho, regParam)

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    println("Finished!!!!!!!!!!!!!!!!!!!!!!!")


    Vectors.fromBreeze(primalConsensus)

  }

}

