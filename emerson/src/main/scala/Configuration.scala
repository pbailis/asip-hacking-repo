package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD



class EmersonParams extends Serializable {
  var eta_0 = 0.01
  var tol = 1.0e-5
  var workerTol = 1.0e-5
  var maxIterations = 1000
  var maxWorkerIterations = 1000
  var miniBatchSize = 100
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
  var learningT = false

  //  var admmRegFactor = 1.0

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
      //  "admmRegFactor" -> admmRegFactor
    )
  }
  override def toString = {
    "{" + toMap.iterator.map {
      case (k,v) => "\"" + k + "\": " + v
    }.toArray.mkString(", ") + "}"
  }
}
