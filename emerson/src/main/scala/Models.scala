package edu.berkeley.emerson

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.rdd.RDD


abstract trait EmersonOptimizer extends Serializable {
  def initialize(params: EmersonParams,
                 lossFunction: LossFunction, regularizationFunction: Regularizer,
                 initialWeights: BV[Double], rawData: RDD[Array[(Double, BV[Double])]])
  def optimize(): BV[Double]

  def statsMap(): Map[String, String]
}


abstract trait BasicEmersonOptimizer extends EmersonOptimizer {
  var params: EmersonParams = null
  var lossFunction: LossFunction = null
  var regularizationFunction: Regularizer = null
  var initialWeights: BV[Double] = null
  var data: RDD[Array[(Double, BV[Double])]] = null
  var nData: Int = 0
  var nSubProblems: Int = 0
  var nDim: Int = 0



  def initialize(params: EmersonParams,
                 lossFunction: LossFunction, regularizationFunction: Regularizer,
                 initialWeights: BV[Double], data: RDD[Array[(Double, BV[Double])]]) {
    println(params)

    this.data = data
    this.params = params
    this.lossFunction = lossFunction
    this.regularizationFunction = regularizationFunction
    this.initialWeights = initialWeights

    nDim = initialWeights.size
    nSubProblems = data.partitions.length

    data.cache()
    val perNodeData = data.map( a => a.length ).collect
    nData = perNodeData.sum

    println(s"Per node data size: ${perNodeData.mkString(",")}")

  }

}

class EmersonModel(val params: EmersonParams,
                   val lossFunction: LossFunction,
                   val regularizationFunction: Regularizer,
                   val optimizer: EmersonOptimizer)
  extends Serializable {

  var initTimeMS = 0L
  var runtimeMS = 0L

  var weights: BV[Double] = null

  def fit(params: EmersonParams, initialWeights: BV[Double], data: RDD[Array[(Double, BV[Double])]]): Unit = {

    val initStartTime = System.currentTimeMillis()
    optimizer.initialize(params, lossFunction, regularizationFunction, initialWeights, data)
    initTimeMS = System.currentTimeMillis() - initStartTime

    val optStartTime = System.currentTimeMillis()
    weights = optimizer.optimize()
    runtimeMS = System.currentTimeMillis() - optStartTime

  }

  def score(data: RDD[Array[(Double, BV[Double])]]): (Double, Double, Double, Double) = {
    assert(weights != null)
    val w = weights // make a local reference to prevent closure capture
    val (nData, totalError) = data.map { data =>
      val totalError = data.view.map{
        case (y, x) =>
          assert(y == 0.0 || y == 1.0)
          val error = if (lossFunction.predict(w, x) != y) { 1.0 } else { 0.0 }
          error
      }.sum
      (data.length.toDouble, totalError)
    }.reduce( (a,b) => (a._1 + b._1, a._2 + b._2) )

    val loss = data.map(d => lossFunction(w, d)).reduce( _ + _ ) / nData.toDouble

    val propError = totalError / nData
    val reg = regularizationFunction(w, params.regParam)
    val objective = loss + reg
    (objective, propError, loss, reg)
  }
}



//
//
///**
// * Train a Support Vector Machine (SVM) using Stochastic Gradient Descent.
// * NOTE: Labels used in SVM should be {0, 1}.
// */
//class SVMWithADMM(val params: EmersonParams) extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
//
//  override val optimizer: ADMM = new ADMM(params, new HingeLoss(), new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new SVMModel(weights, intercept)
//  }
//}
//
//
//class SVMWithAsyncADMM(val params: EmersonParams) extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
//
//  override val optimizer = new AsyncADMM(params, new HingeLoss(), new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new SVMModel(weights, intercept)
//  }
//}
//
//
//class SVMWithHOGWILD(val params: EmersonParams) extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
//
//  override val optimizer = new HOGWILDSGD(params, new HingeLoss(), new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new SVMModel(weights, intercept)
//  }
//}
//
//
///**
// * Train a Support Vector Machine (SVM) using Stochastic Gradient Descent.
// * NOTE: Labels used in SVM should be {0, 1}.
// */
//class LRWithADMM(val params: EmersonParams)
//  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
//
//  override val optimizer: ADMM = new ADMM(params, new LogisticLoss(),
//					  new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new LogisticRegressionModel(weights, intercept)
//  }
//}
//
//class LRWithAsyncADMM(val params: EmersonParams)
//  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
//
//  override val optimizer = new AsyncADMM(params, new LogisticLoss(),
//					 new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new LogisticRegressionModel(weights, intercept)
//  }
//}
//
//class LRWithHOGWILD(val params: EmersonParams)
//  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
//
//  override val optimizer = new HOGWILDSGD(params, new LogisticLoss(),
//					  new L2Regularizer())
//
//  //override protected val validators = List(DataValidators.binaryLabelValidator)
//  override protected def createModel(weights: Vector, intercept: Double) = {
//    new LogisticRegressionModel(weights, intercept)
//  }
//}
