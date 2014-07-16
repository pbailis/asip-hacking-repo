package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * AN ADMM Local Solver is used to solve the optimization problem within each partition.
 */
@DeveloperApi
trait LocalOptimizer extends Serializable {
  def apply(nSubProblems: Int, data: Array[(Double, Vector)],
            w: BV[Double], w_avg: BV[Double], lambda: BV[Double],
            rho: Double): BV[Double]
}


//@DeveloperApi
//class GradientDescentLocalOptimizer(val gradient: Gradient,
//                                    val eta_0: Double = 1.0,
//                                    val maxIterations: Int = Integer.MAX_VALUE,
//                                    val epsilon: Double = 0.001) extends LocalOptimizer {
//  def apply(data: Array[(Double, Vector)], w0: BV[Double], w_avg: BV[Double], lambda: BV[Double],
//            rho: Double): BV[Double] = {
//    var t = 0
//    var residual = Double.MaxValue
//    val w = w0.copy
//    val nExamples = data.length
//    while(t < maxIterations || residual > epsilon) {
//      // Compute the total gradient (and loss)
//      val (gradientSum, lossSum) =
//        data.foldLeft((w * 0.0, 0.0)) { (c, v) =>
//          val (gradSum, lossSum) = c
//          val (label, features) = v
//          val loss = gradient.compute(features, label, Vectors.fromBreeze(w), Vectors.fromBreeze(gradSum))
//          (gradSum, lossSum + loss)
//        }
//      // compute the gradient of the full lagrangian
//      val gradL = (gradientSum / nExamples.toDouble) + lambda + (w - w_avg) * rho
//      // Set the learning rate
//      val eta_t = eta_0 / (t + 1)
//      // w = w + eta_t * point_gradient
//      axpy(-eta_t, gradL, w)
//      // Compute residual
//      residual = eta_t * norm(gradL, 2.0)
//      t += 1
//    }
//    // Check the local prediction error:
//    val propCorrect =
//      data.map { case (y,x) => if (x.toBreeze.dot(w) * (y * 2.0 - 1.0) > 0.0) 1 else 0 }
//        .reduce(_ + _).toDouble / nExamples.toDouble
//    println(s"Local prop correct: $propCorrect")
//    println(s"Local iterations: ${t}")
//    // Return the final weight vector
//    w
//  }
//}

@DeveloperApi
class SGDLocalOptimizer(val gradient: Gradient, val updater: Updater) extends LocalOptimizer {

  var eta_0: Double = 1.0
  var maxIterations: Int = Integer.MAX_VALUE
  var epsilon: Double = 0.001
  var miniBatchFraction: Double = 0.1

  def apply(nSubProblems: Int, data: Array[(Double, Vector)],
            w0: BV[Double], w_avg: BV[Double], lambda: BV[Double],
            rho: Double): BV[Double] = {
    val nExamples = data.length
    val dim = data(0)._2.size
    val miniBatchSize = math.min(nExamples, (miniBatchFraction * nExamples).toInt)
    val rnd = new java.util.Random()

    var residual = 1.0
    var w = w0.copy
    var gradSum = BV.zeros[Double](dim)
    var t = 0
    while(t < maxIterations || residual > epsilon) {
      gradSum *= 0.0 // Clear the gradient sum
      for (b <- 0 until miniBatchSize) {
        val ind = if (miniBatchSize == nExamples) b else rnd.nextInt(nExamples)
        val (label, features) = data(ind)
        gradient.compute(features, label, Vectors.fromBreeze(w), Vectors.fromBreeze(gradSum))
      }
      // gradSum += rho * (lambda + w - w_avg)
      axpy(rho, (lambda + w - w_avg), gradSum )

      // Set the learning rate
      val eta_t = eta_0 / (t + 1.0)
      // w = w + eta_t * point_gradient
      axpy(-eta_t, gradL, w)
      // Compute residual
      residual = (eta_t * norm(gradL, 2.0) + residual) / 2.0
      t += 1
    }
    // Check the local prediction error:
    val propCorrect =
      data.map { case (y,x) => if (x.toBreeze.dot(w) * (y * 2.0 - 1.0) > 0.0) 1 else 0 }
        .reduce(_ + _).toDouble / nExamples.toDouble
    println(s"Local prop correct: $propCorrect")
    println(s"Local iterations: ${t}")
    // Return the final weight vector
    w
  }
}


class ADMM(var localOptimizer: LocalOptimizer,
    private var numIterations: Int = 100,
    private var regParam: Double = 0.0,
    private var epsilon: Double = 1.0e-5) extends Optimizer with Logging {

  /**
   * Set the number of iterations for ADMM. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the local optimizer to use for subproblems.
   */
  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }


  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    val blockData: RDD[Array[(Double, Vector)]] = data.mapPartitions(iter => Iterator(iter.toArray)).cache()
    val dim = blockData.map(block => block(0)._2.size).first()
    val nExamples = blockData.map(block => block.length).reduce(_+_)
    val numPartitions = blockData.partitions.length
    println(s"nExamples: $nExamples")
    println(s"dim: $dim")
    println(s"number of solver ${numPartitions}")


    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue
    var iter = 0
    var rho  = 0.0

    // Make a zero vector
    var wAndLambda = blockData.map{ block =>
      val dim = block(0)._2.size
      (BV.zeros[Double](dim), BV.zeros[Double](dim))
    }
    var w_avg = BV.zeros[Double](dim)

    val optimizer = localOptimizer
    while (iter < numIterations || primalResidual > epsilon || dualResidual > epsilon) {
      println(s"Starting iteration ${iter}.")
      // Compute w and new lambda
      wAndLambda = blockData.zipPartitions(wAndLambda) { (dataIter, modelIter) =>
        dataIter.zip(modelIter).map { case (data, (w_old, lambda_old)) =>
          // Update the lagrangian Multiplier by taking a gradient step
          val lambda = lambda_old + (w_old - w_avg) * rho
          val w = optimizer(data, w_old, w_avg, lambda, rho)
          (w, lambda)
        }
      }.cache()
      // Compute new w_avg
      val new_w_avg = blockData.zipPartitions(wAndLambda) { (dataIter, modelIter) =>
        dataIter.zip(modelIter).map { case (data, (w, _)) => w * data.length.toDouble }
      }.reduce(_ + _) / nExamples.toDouble
      println(s"new w_avg: $new_w_avg")

      // Update the residuals
      // primalResidual = sum( ||w_i - w_avg||_2^2 )
      primalResidual = Math.pow( wAndLambda.map { case (w, _) => norm(w - new_w_avg, 2.0) }.reduce(_ + _), 2)
      dualResidual = rho * Math.pow(norm(new_w_avg - w_avg, 2.0), 2)

      // Rho upate from Boyd text
      if (rho == 0.0) {
        rho = epsilon
      } else if (primalResidual > 10.0 * dualResidual) {
        rho = 2.0 * rho
        println(s"Increasing rho: $rho")
      } else if (dualResidual > 10.0 * primalResidual) {
        rho = rho / 2.0
        println(s"Decreasing rho: $rho")
      }

      w_avg = new_w_avg

      println(s"Iteration: ${iter}")
      println(s"(Primal Resid, Dual Resid, Rho): ${primalResidual}, \t ${dualResidual}, \t ${rho}")

      iter += 1
    }

    Vectors.fromBreeze(w_avg)
  }

}
