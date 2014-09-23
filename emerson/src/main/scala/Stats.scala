package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD




object Stats {
  def apply(primalVar: BV[Double], dualVar: BV[Double],
    msgsSent: Int = 0,
    msgsRcvd: Int = 0,
    localIters: Int = 0,
    sgdIters: Int = 0,
    dualUpdates: Int = 0,
    residual: Double = 0.0,
    dataSize: Int = 0) = {
    new Stats(
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


case class Stats(
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
    Stats(null, null,
      msgsSent = msgsSent,
      msgsRcvd = msgsRcvd,
      localIters = localIters,
      sgdIters = sgdIters,
      dualUpdates = dualUpdates,
      dataSize = dataSize,
      residual = residual,
      nWorkers = nWorkers)
  }

  def +(other: Stats) = {
    new Stats(
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

  def toStringShort = {
    "{" + toMap.iterator.filter { 
      case (k,v) => k != "primalAvg" && k != "dualAvg"
    }.map {
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

