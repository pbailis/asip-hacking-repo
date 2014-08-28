package edu.berkeley.emerson

import java.util.concurrent.TimeUnit
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


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

