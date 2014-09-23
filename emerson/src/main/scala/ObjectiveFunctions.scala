package edu.berkeley.emerson

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}



trait LossFunction extends Serializable {

  def predict(w: BV[Double], x: BV[Double]): Double = {
    if(w.dot(x) > 0.0) { 1.0 } else { 0.0 }
  }

  /**
   * Add gradient for point to running sum.
   */
  def addGradient(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double

  /**
   * Add gradient for point to running sum.
   */
  def addGradient(w: BV[Double], data: Array[(Double, BV[Double])], cumGrad: BV[Double]): Double = {
    var i = 0
    var sum = 0.0
    while (i < data.length) {
      sum += addGradient(w, data(i)._2, data(i)._1, cumGrad)
      i += 1
    }
    sum
  }



  /**
   * Evaluate the loss at a point
   */
  def apply(w: BV[Double], x: BV[Double], y: Double): Double = 
    addGradient(w, x, y, null)


  /**
   * Evaluate the loss on a collection of data points
   */
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


/**
 * The Hinge Loss for SVM
 */
class HingeLoss extends LossFunction {

  override def addGradient(w: BV[Double], x: BV[Double], y: Double, cumGrad: BV[Double]): Double = {
    assert(y == 0.0 || y == 1.0)
    val yscaled = 2.0 * y - 1.0
    val wdotx = w.dot(x)
    if (yscaled * wdotx < 1.0) {
      if (cumGrad != null) { axpy(-yscaled, x, cumGrad) }
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
class LogisticLoss extends LossFunction {

  /**
   * Surprisingly this function is stable for extremely small and large x
   */
  def sigmoid(x: Double) = 1.0 / (1.0 + math.exp(-x)) 

  override def addGradient(w: BV[Double], x: BV[Double], label: Double, 
			   cumGrad: BV[Double]) = {
    val wdotx = w.dot(x)
    val gradientMultiplier = label - sigmoid(wdotx)
    // Note we negate the gradient here since we ant to minimize the negative of the likelihood
    if (cumGrad != null) { axpy(-gradientMultiplier, x, cumGrad) }
    val margin = -1.0 * wdotx
    val logExpMargin = // mathematically stable approximation of log(1 + exp(w dot x))
      if (margin > 20) { 
        margin
      } else if (margin < -20) {
        0
      } else {
        math.log1p(math.exp(margin))
      }

    val logLikelihood =
      if (label > 0) {
        -logExpMargin // log1p(x) = log(1+x)
      } else {
        margin - logExpMargin
      }
    -logLikelihood
  }
}


trait Regularizer extends Serializable {

  /**
   * Evaluate the cost of the regularizer at a point
   */
  def apply(w: BV[Double], regParam: Double): Double =
    addGradient(w, regParam, null)

  /**
   * Add the gradient of the regularizer
   * @param w
   * @param regParam
   * @param cumGrad
   * @return
   */
  def addGradient(w: BV[Double], regParam: Double, cumGrad: BV[Double]): Double

  def consensus(primalAvg: BV[Double], dualAvg: BV[Double], 
		nSolvers: Int, rho: Double, 
		regParam: Double): BV[Double]

  def gradientStep(w: BV[Double], regParam: Double, stepSize: Double, 
		   cumGrad: BV[Double])

  def loss(w: BV[Double], regParam: Double): Double 
 
}


/*
0 & = \nabla_z \left( \lambda ||z||_2^2 + \sum_{i=1}^N \left( \mu_i^T (x_i - z) +  \frac{\rho}{2} ||x_i - z||_2^2 \right)  \right) \\
0 & = \lambda z - \sum_{i=1}^N \left(\mu_i + \rho (x_i - z) \right)  \\
0 & =\lambda z - N \bar{u} - \rho N \bar{x} + \rho N z    \\
0 & = z (\lambda + \rho N) -  N (\bar{u} + \rho \bar{x} )  \\
z & = \frac{ N}{\lambda + \rho N} (\bar{u} + \rho \bar{x})
z & = \frac{ \rho N}{\lambda + \rho N} (\frac{1}{\rho}\bar{u} + \bar{x})
*/
class L2Regularizer extends Regularizer {

  override def addGradient(w: BV[Double], regParam: Double, cumGrad: BV[Double]): Double = {
    if (cumGrad != null) { axpy(regParam, w, cumGrad) }
    math.pow(norm(w,2),2) * (regParam / 2.0)
  }

  override def consensus(primalAvg: BV[Double], dualAvg: BV[Double], 
			 nSolvers: Int, rho: Double, regParam: Double): BV[Double] = {
    if (rho == 0.0) {
      primalAvg
    } else {
      val multiplier = (nSolvers * rho) / (regParam + nSolvers * rho)
      println(s"multiplier: $multiplier")
      (primalAvg + dualAvg / rho) * multiplier
    }
    //primalAvg.copy
  }

  
  override def gradientStep(w: BV[Double], regParam: Double, stepSize: Double, 
			    cumGrad: BV[Double]) {
    // Add the regParam to the cumulative gradient
    axpy(regParam, w, cumGrad)
    // Substract the gradient from w
    axpy(-stepSize, cumGrad, w)
  }

  def loss(w: BV[Double], regParam: Double): Double =
    (regParam / 2.0) * math.pow(norm(w, 2), 2)

}


class L1Regularizer extends Regularizer {

  def softThreshold(alpha: Double, x: BV[Double]): BV[Double] = {
    val ret = BV.zeros[Double](x.size)
    softThreshold(alpha, x, ret)
    ret
  }

 def softThreshold(alpha: Double, x: BV[Double], out: BV[Double]) {
    var i = 0
    while (i < x.size) {
      if(x(i) < alpha) {
        out(i) = x(i) + alpha
      } else if (x(i) > alpha) {
        out(i) = x(i) - alpha
      } else {
      	out(i) = 0.0
      }
      i += 1
    }
  }

  override def addGradient(w: BV[Double], regParam: Double, cumGrad: BV[Double]): Double = {
    if(cumGrad != null) {
      var i = 0
      while (i < w.size) {
        cumGrad(i) += math.signum(w(i)) * regParam
        i += 1
      }
    }
    regParam * norm(w, 1)
  }


  override def consensus(primalAvg: BV[Double], dualAvg: BV[Double], 
			 nSolvers: Int, rho: Double, regParam: Double): BV[Double] = {
    if (rho == 0.0) {
      softThreshold(regParam, primalAvg)
    } else {
      val threshold = regParam / (nSolvers * rho)
      softThreshold(threshold, primalAvg + dualAvg / rho)
    }
  }

  override def gradientStep(w: BV[Double], regParam: Double, stepSize: Double, 
			    cumGrad: BV[Double]) {
    // Subtract the gradeint and threshold
    axpy(-stepSize, cumGrad, w)
    val threshold = regParam
    softThreshold(threshold, w, w)
  }

  def loss(w: BV[Double], regParam: Double): Double =
    (regParam / 2.0) * norm(w, 1)

}
