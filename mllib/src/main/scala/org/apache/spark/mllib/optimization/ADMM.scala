package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


trait FastGradient extends Serializable {
  def apply(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double])
}

class FastHingeGradient extends FastGradient {
  override def apply(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]) {
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      axpy(-yscaled, x, cumGrad)
    }
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

@DeveloperApi
class SGDLocalOptimizer(val subProblemId: Int,
                        val data: Array[(Double, BV[Double])],
                        var primalVar: BV[Double],
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

  var dualVar = BV.zeros[Double](dim)

  def dualUpdate(primalConsensus: BV[Double], rho: Double) {
    // Do the dual update
    dualVar += (primalVar - primalConsensus) * rho
  }

  def primalUpdate(primalConsensus: BV[Double], rho: Double) {
    var t = 0
    residual = Double.MaxValue
    while(t < maxIterations && residual > epsilon) {
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

  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(rawData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    val primal0 = initialWeights.toBreeze

    val solvers: RDD[SGDLocalOptimizer] =
      rawData.mapPartitionsWithIndex { (ind, iter) =>
        val data: Array[(Double, BV[Double])] = iter.map { case (label, features) => (label, features.toBreeze)}.toArray
        val solver = new SGDLocalOptimizer(ind, data, primal0.copy, gradient,
          eta_0 = eta_0, epsilon = localEpsilon, maxIterations = localMaxIterations, miniBatchSize = miniBatchSize)
        Iterator(solver)
      }.cache()

    val nDim = initialWeights.size
    val nExamples: Int = solvers.map(s => s.data.length).reduce(_+_)
    val nSolvers = solvers.partitions.length


    println(s"nExamples: $nExamples")
    println(s"dim: $nDim")
    println(s"number of solver $nSolvers")

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue
    var iteration = 0
    var rho  = 0.0

    var primalConsensus = initialWeights.toBreeze.copy

    println(s"ADMM numIterations: $numIterations")
    while (iteration < numIterations && (primalResidual > epsilon || dualResidual > epsilon) ) {
      println("========================================================")
      println(s"Starting iteration $iteration.")
      var (primalAvg, dualAvg) = solvers.map{ solver =>
        // Do a dual update
        solver.dualUpdate(primalConsensus, rho)
        // Do a primal update
        solver.primalUpdate(primalConsensus, rho)
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
      if (rho == 0.0) {
        rho = 1.0
      } else if (primalResidual > 10.0 * dualResidual) {
        rho = 2.0 * rho
        println(s"Increasing rho: $rho")
      } else if (dualResidual > 10.0 * primalResidual) {
        rho = rho / 2.0
        println(s"Decreasing rho: $rho")
      }

      println(s"Iteration: $iteration")
      println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")

      iteration += 1
    }

    println(s"${primalConsensus.toArray.mkString(",")}")

    Vectors.fromBreeze(primalConsensus)
  }

}

