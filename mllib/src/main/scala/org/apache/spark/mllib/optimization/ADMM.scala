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
    if (rho == 0.0) {
      softThreshold(regParam, primalAvg)
    } else {
      // Joey: rederive this equation:
      softThreshold(regParam / nSolvers.toDouble, primalAvg * rho + dualAvg)
    }
  }
}


@DeveloperApi
class SGDLocalOptimizer(val subProblemId: Int,
                        val data: Array[(Double, BV[Double])],
                        @volatile var primalVar: BV[Double],
                        val gradient: FastGradient,
                        val eta_0: Double,
                        val epsilon: Double,
                        val maxIterations: Int,
                        val miniBatchSize: Int
                        ) extends Serializable with Logging {

  val nExamples = data.length
  val dim = data(0)._2.size
  val rnd = new java.util.Random(subProblemId)

  var residual: Double = Double.MaxValue

  var grad = BV.zeros[Double](dim)

  @volatile var dualVar = BV.zeros[Double](dim)

  @volatile var primalConsensus = BV.zeros[Double](dim)

  @volatile var rho = 1.0

  def dualUpdate() {
    // Do the dual update
    dualVar += (primalVar - primalConsensus) * rho
  }

  def primalUpdate(remainingTimeMS: Long = Long.MaxValue) {
    sgd(remainingTimeMS)
  }

  def lbfgs(remainingTimeMS: Long = Long.MaxValue) {
    val lbfgs = new breeze.optimize.LBFGS[BDV[Double]](maxIterations, 3)
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
        cumGrad += dualVar
        axpy(rho, x - primalConsensus, cumGrad)
        (obj, cumGrad)
      }
    }
    primalVar = lbfgs.minimize(f, primalConsensus.toDenseVector)
  }


  def sgd(remainingTimeMS: Long = Long.MaxValue) {
    var t = 0
    residual = Double.MaxValue
    val startTime = System.currentTimeMillis()
    var currentTime = startTime
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

      if (t % 1000 == 0) {
        currentTime = System.currentTimeMillis()
      }

      t += 1
    }
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

  def setup(rawData: RDD[(Double, Vector)], initialWeights: Vector) {
    val primal0 = initialWeights.toBreeze
    solvers =
      rawData.mapPartitionsWithIndex { (ind, iter) =>
        val data: Array[(Double, BV[Double])] = iter.map { case (label, features) => (label, features.toBreeze)}.toArray
        val solver = new SGDLocalOptimizer(ind, data, primal0.copy, gradient,
          eta_0 = eta_0, epsilon = localEpsilon, maxIterations = localMaxIterations, miniBatchSize = miniBatchSize)
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
    val nExamples: Int = solvers.map(s => s.data.length).reduce(_+_)
    val nSolvers = solvers.partitions.length


    println(s"nExamples: $nExamples")
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
      var (primalAvg, dualAvg) = solvers.map{ solver =>
        // Do a dual update
        solver.primalConsensus = primalConsensus
        solver.rho = rho
        solver.dualUpdate()
        // Do a primal update
        solver.primalUpdate(timeRemaining)
        // Return the scaled primal and dual values
        (solver.primalVar * solver.data.length.toDouble, solver.dualVar * solver.data.length.toDouble)
        }
        .reduce( (a,b) => (a._1 + b._1, a._2 + b._2) )
      primalAvg /= nExamples.toDouble
      dualAvg /= nExamples.toDouble

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      primalConsensus = consensus(primalAvg, dualAvg, nSolvers, rho, regParam)

      // Compute the residuals
      primalResidual = solvers.map(s => norm(s.primalVar - primalConsensus, 2) * s.data.length).reduce(_+_) / nExamples.toDouble
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

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)


    Vectors.fromBreeze(primalConsensus)
  }

}

