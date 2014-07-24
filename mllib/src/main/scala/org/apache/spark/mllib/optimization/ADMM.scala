package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * AN ADMM Local Solver is used to solve the optimization problem within each partition.
 */
@DeveloperApi
trait LocalOptimizer extends Serializable {
  def apply(nSubProblems: Int, data: Array[(Double, Vector)],
            w: BV[Double], w_consensus: BV[Double], dualVar: BV[Double],
            rho: Double, regParam: Double): (BV[Double], String)
}




    //@DeveloperApi
//class GradientDescentLocalOptimizer(val gradient: Gradient,
//                                    val eta_0: Double = 1.0,
//                                    val maxIterations: Int = Integer.MAX_VALUE,
//                                    val epsilon: Double = 0.001) extends LocalOptimizer {
//  def apply(data: Array[(Double, Vector)], w0: BV[Double], w_avg: BV[Double], dualVar: BV[Double],
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
//      val gradL = (gradientSum / nExamples.toDouble) + dualVar + (w - w_avg) * rho
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
class SGDLocalOptimizer(val gradient: Gradient, val updater: Updater) extends LocalOptimizer with Logging {

  var eta_0: Double = 1.0
  var maxIterations: Int = Integer.MAX_VALUE
  var epsilon: Double = 0.001
  var miniBatchFraction: Double = 0.1

  def apply(nSubProblems: Int, data: Array[(Double, Vector)],
            w0: BV[Double], w_consensus: BV[Double], dualVar: BV[Double],
            rho: Double, regParam: Double): (BV[Double], String) = {
    val nExamples = data.length
    val dim = data(0)._2.size
    val miniBatchSize = math.max(1, math.min(nExamples, (miniBatchFraction * nExamples).toInt))
    val rnd = new java.util.Random()

    var residual = 1.0
    var w = w0.copy
    var grad = BV.zeros[Double](dim)
    var t = 0
    while(t < maxIterations && residual > epsilon) {
      grad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize < nExamples) rnd.nextInt(nExamples) else b
        val (label, features) = data(ind)
        gradient.compute(features, label, Vectors.fromBreeze(w), Vectors.fromBreeze(grad))
        b += 1
      }
      // Normalize the gradient to the batch size
      grad /= miniBatchSize.toDouble
      // Add the lagrangian + augmenting term.
      // grad += rho * (dualVar + w - w_avg)
      axpy(rho, dualVar + w - w_consensus, grad)
      // Set the learning rate
      val eta_t = eta_0 / (t.toDouble + 1.0)
      // w = w + eta_t * point_gradient
      // axpy(-eta_t, grad, w)
      val wOld = w.copy
      w = updater.compute(Vectors.fromBreeze(w), Vectors.fromBreeze(grad), eta_t, 1, regParam)._1.toBreeze
      // Compute residual.  This is a decaying residual definition.
      // residual = (eta_t * norm(grad, 2.0) + residual) / 2.0
      residual = norm(w - wOld, 2.0)
      if((t % 10) == 0) {
        logInfo(s"Residual: $residual ")
      }
      t += 1
    }
    // Compute the total gradient and total loss and proportion of correct examples locally
    grad *= 0.0
    var i = 0
    var loss = 0.0
    var correct = 0
    val vectorW = Vectors.fromBreeze(w)
    while (i < nExamples) {
      val (y, x) = data(i)
      if (x.toBreeze.dot(w) * (y * 2.0 - 1.0) > 0.0) correct += 1
      loss += gradient.compute(x, y, vectorW, Vectors.fromBreeze(grad))
      i += 1
    }
    val propCorrect = correct / nExamples.toDouble
    val gradientNorm = norm(grad, 2.0)
    logInfo(s"t = $t and residual = $residual")
    logInfo(s"Local prop correct: $propCorrect")
    val stats = s"STATS: t = $t, residual = $residual, accuracy = $propCorrect, " +
      s"gradientNorm = $gradientNorm, loss = $loss"
    // Return the final weight vector
    (w, stats)
  }
}


class ADMM(var localOptimizer: LocalOptimizer) extends Optimizer with Logging {

  var numIterations: Int = 100
  var regParam: Double = 1.0
  var epsilon: Double = 1.0e-5

  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(rawData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    val blockData: RDD[Array[(Double, Vector)]] = rawData.mapPartitions(iter => Iterator(iter.toArray)).cache()
    val dim = blockData.map(block => block(0)._2.size).first()
    val nExamples = blockData.map(block => block.length).reduce(_+_)
    val numPartitions = blockData.partitions.length
    val localReg = regParam / numPartitions.toDouble 
    println(s"nExamples: $nExamples")
    println(s"dim: $dim")
    println(s"number of solver $numPartitions")

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue
    var iteration = 0
    var rho  = 0.0

    // Make a zero vector
    var wAndDualVar = blockData.map{ block =>
      val dim = block(0)._2.size
      (BV.zeros[Double](dim), BV.zeros[Double](dim))
    }
    var w_avg = BV.zeros[Double](dim)

    println(s"ADMM numIterations: $numIterations")

    val optimizer = localOptimizer
    while (iteration < numIterations && (primalResidual > epsilon || dualResidual > epsilon) ) {
      println("========================================================")
      println(s"Starting iteration $iteration.")
      // Compute w and new dualVar
      val tmp = blockData.zipPartitions(wAndDualVar) { (dataIterator, modelIterator) =>
        dataIterator.zip(modelIterator).map { case (data, (w_old, dualVar_old)) =>
          // Update the lagrangian Multiplier by taking a gradient step
          val dualVar = dualVar_old + (w_old - w_avg)
          val (w, stats) = optimizer(numPartitions, data, w_old, w_avg, dualVar, rho, localReg)
          (w, dualVar, stats)
        }
      }.cache()
      wAndDualVar = tmp.map { case (w, dualVar, stats) => (w, dualVar) }
      tmp.map { case (w, dualVar, stats) => stats } .collect.foreach(println(_))

      // Compute new w_avg
      val new_w_avg = blockData.zipPartitions(wAndDualVar) { (dataIterator, modelIterator) =>
        dataIterator.zip(modelIterator).map { case (data, (w, _)) => w * data.length.toDouble }
      }.reduce(_ + _) / nExamples.toDouble
      println(s"new w_avg:(${new_w_avg.toArray.min}, ${new_w_avg.toArray.max})")


      // Update the residuals
      // primalResidual = sum( ||w_i - w_avg||_2^2 )
      primalResidual = wAndDualVar.map { case (w, _) => Math.pow(norm(w - new_w_avg, 2.0),2) }.reduce(_ + _) /
        numPartitions.toDouble
      dualResidual = Math.pow(norm(new_w_avg - w_avg, 2.0), 2)

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

      w_avg = new_w_avg
      println(s"Iteration: $iteration")
      println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")

      iteration += 1
    }

    println(s"${w_avg.toArray.mkString(",")}")

    Vectors.fromBreeze(w_avg)
  }

}


//@DeveloperApi
//trait ConsensusRegularizer extends Serializable {
//  def getUpdater: Updater
//
//  def apply(w_avg: BV[Double], dualVar: BV[Double],
//            rho: Double, regParam: Double, nData: Int): BV[Double]
//}
//
//
//@DeveloperApi
//class AvgConsensuRegularizer extends ConsensusRegularizer {
//  override def getUpdater: Updater = new SimpleUpdater()
//
//  override def apply(w_avg: BV[Double], dualVar: BV[Double],
//                     rho: Double, regParam: Double, nData: Int): BV[Double] = {
//    w_avg
//  }
//}
//
//
//@DeveloperApi
//class L2ConsensusRegularizer extends ConsensusRegularizer {
//  override def getUpdater: Updater = new SquaredL2Updater()
//
//  override def apply(w_avg: BV[Double], dualVar: BV[Double],
//                     rho: Double, regParam: Double, nData: Int): BV[Double] = {
//    (w_avg + dualVar) * (rho / ((1.0 / regParam) + nData.toDouble * rho))
//  }
//}
//
//@DeveloperApi
//class L1ConsensusRegularizer extends ConsensusRegularizer {
//  override def getUpdater: Updater = new L1Updater()
//
//  override def apply(w_avg: BV[Double], dualVar: BV[Double],
//                     rho: Double, regParam: Double, nData: Int): BV[Double] = {
//    (w_avg + dualVar) * (rho / ((1.0 / regParam) + nData.toDouble * rho))
//  }
//}


//class ADMMConsensusRegularizer(var localOptimizer: LocalOptimizer,
//           var consensusUpdater: ConsensusRegularizer) extends Optimizer with Logging {
//  var numIterations: Int = 100
//  var regParam: Double = 1.0
//  var epsilon: Double = 1.0e-5
//
//  /**
//   * Solve the provided convex optimization problem.
//   */
//  override def optimize(rawData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
//
//    val blockData: RDD[Array[(Double, Vector)]] = rawData.mapPartitions(iter => Iterator(iter.toArray)).cache()
//    val dim = blockData.map(block => block(0)._2.size).first()
//    val nExamples = blockData.map(block => block.length).reduce(_+_)
//    val numPartitions = blockData.partitions.length
//    val localReg = 0.5 / numPartitions.toDouble * regParam
//    val globalReg = 0.5 * regParam
//    println(s"nExamples: $nExamples")
//    println(s"dim: $dim")
//    println(s"number of solver $numPartitions")
//
//
//    var primalResidual = Double.MaxValue
//    var dualResidual = Double.MaxValue
//    var iteration = 0
//    var rho  = 0.0
//
//
//    // Make a zero vector
//    var wAnddualVar = blockData.map{ block =>
//      val dim = block(0)._2.size
//      (BV.zeros[Double](dim), BV.zeros[Double](dim))
//    }
//    var w_avg = BV.zeros[Double](dim)
//
//    val optimizer = localOptimizer
//    while (iteration < numIterations && (primalResidual > epsilon || dualResidual > epsilon) ) {
//      println(s"Starting iteration $iteration.")
//      // Compute w and new dualVar
//      wAnddualVar = blockData.zipPartitions(wAnddualVar) { (dataIterator, modelIterator) =>
//        dataIterator.zip(modelIterator).map { case (data, (w_old, dualVar_old)) =>
//          // compute the new consenus variable value
//          val w_consensus = consensusUpdater(w_avg, dualVar_old, rho, globalReg, nExamples)
//          // Update the lagrangian Multiplier by taking a gradient step
//          val dualVar = dualVar_old + (w_old - w_consensus)
//          val w = optimizer(numPartitions, data, w_old, w_consensus, dualVar, rho, localReg)
//          (w, dualVar)
//        }
//      }.cache()
//
//      // Compute new w_avg
//      val new_w_avg = blockData.zipPartitions(wAnddualVar) { (dataIterator, modelIterator) =>
//        dataIterator.zip(modelIterator).map { case (data, (w, _)) => w * data.length.toDouble }
//      }.reduce(_ + _) / nExamples.toDouble
//      println(s"new w_avg: $w_avg")
//
//
//      // Update the residuals
//      // primalResidual = sum( ||w_i - w_avg||_2^2 )
//      primalResidual = wAnddualVar.map { case (w, _) => Math.pow(norm(w - new_w_avg, 2.0),2) }.reduce(_ + _) /
//        numPartitions.toDouble
//      dualResidual = rho * Math.pow(norm(new_w_avg - w_avg, 2.0), 2)
//
//      // Rho upate from Boyd text
//      if (rho == 0.0) {
//        rho = epsilon
//      } else if (primalResidual > 10.0 * dualResidual) {
//        rho = 2.0 * rho
//        println(s"Increasing rho: $rho")
//      } else if (dualResidual > 10.0 * primalResidual) {
//        rho = rho / 2.0
//        println(s"Decreasing rho: $rho")
//      }
//
//      w_avg = new_w_avg
//
//      println(s"Iteration: $iteration")
//      println(s"(Primal Resid, Dual Resid, Rho): $primalResidual, \t $dualResidual, \t $rho")
//
//      iteration += 1
//    }
//
//    Vectors.fromBreeze(w_avg)
//  }
//
//}
//
