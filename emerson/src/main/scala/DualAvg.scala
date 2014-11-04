package edu.berkeley.emerson

import java.util.UUID
import java.util.concurrent._
import java.util.concurrent.atomic._

import akka.actor._
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg.{norm, axpy, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.Logging
import org.apache.spark.deploy.worker.Worker
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

//
//case class AsyncSubProblem(data: Array[(Double, Vector)], comm: WorkerCommunication)

object DAInternalMessages {
  class WakeupMsg
  class PingPong
  class DeltaUpdate(val firstSend: Boolean,
                    val delta: BV[Double])
}

class DAWorkerCommunicationHack {
  var ref: DAWorkerCommunication = null
}

object DASetupBlock {
  var initialized = false
  val workers = new Array[DualAvgSGDWorker](128)

}


class DAWorkerCommunication(val address: String, val hack: DAWorkerCommunicationHack) extends Actor with Logging {
  hack.ref = this
  val others = new mutable.HashMap[Int, ActorSelection]
  var selfID: Int = -1

  @volatile var optimizer: DualAvgSGDWorker = null

  def receive = {
    case ppm: InternalMessages.PingPong => {
      logInfo("new message from " + sender)
    }
    case m: InternalMessages.WakeupMsg => {
      logInfo("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: DAInternalMessages.DeltaUpdate => {
      if (optimizer != null) {
        optimizer.dualSum += d.delta
        optimizer.msgsRcvd.getAndIncrement()
        if (d.firstSend) { 
          optimizer.rcvdFrom.getAndIncrement()
        }
      }
    }
    case _ => println("hello, world!")
  }

  def shuttingDown: Receive = {
    case _ => println("GOT SHUTDOWN!")
  }

  def connectToOthers(allHosts: Array[String]) {
    var i = 0
    //logInfo(s"Connecting to others ${allHosts.mkString(",")} ${allHosts.length}")
    for (host <- allHosts) {
      if (!host.equals(address)) {
        //logInfo(s"Connecting to $host, $i")
        others.put(i, context.actorSelection(allHosts(i)))

        implicit val timeout = Timeout(15 seconds)
        val f = others(i).resolveOne()
        Await.ready(f, Duration.Inf)
        logInfo(s"Connected to ${f.value.get.get}")
      } else {
        selfID = i
      }
      i += 1
    }
  }

  def sendPingPongs() {
    for (other <- others.values) {
      other ! new InternalMessages.PingPong
    }
  }

  def broadcastDeltaUpdate(firstSend: Boolean, delta: BV[Double]) {
    val msg = new DAInternalMessages.DeltaUpdate(firstSend, delta)
    for (other <- others.values) {
      other ! msg
    }
  }
}



class DualAvgSGDWorker(subProblemId: Int,
                       nSubProblems: Int,
                       nData: Int,
                       data: Array[(Double, BV[Double])],
                       lossFun: LossFunction,
                       params: EmersonParams,
                       regularizer: Regularizer,
                       val comm: DAWorkerCommunication)
  extends ADMMLocalOptimizer(subProblemId = subProblemId, nSubProblems,
			    nData = nData, data = data, lossFun = lossFun, regularizer, params)
  with Logging {

  comm.optimizer = this

  @volatile var done = false
  @volatile var msgsSent = 0
  @volatile var msgsRcvd = new AtomicInteger()
  @volatile var lastSent: BV[Double] = BV.zeros(nDim)
  @volatile var dualSum: BV[Double] = BV.zeros(nDim)
  @volatile var primalSum: BV[Double] = BV.zeros(nDim)

  val rcvdFrom = new AtomicInteger()

  @volatile var eta_t = params.eta_0

  override def getStats() = {
    Stats(primalVar = primalVar, dualVar = dualVar,
      msgsSent = msgsSent, msgsRcvd = msgsRcvd.get(),
      localIters = localIters,
      dataSize = data.length)
  }

  val broadcastThread = new Thread {
    override def run {
      var firstSend = true
      val startTime = System.currentTimeMillis()
      while (!done) {
        val tmp = dualVar.copy
        comm.broadcastDeltaUpdate(firstSend, tmp - lastSent)
        firstSend = false
        lastSent = tmp
        msgsSent += 1
        Thread.sleep(params.broadcastDelayMS)
        // Check to see if we are done
        done = (System.currentTimeMillis() - startTime) > params.runtimeMS
      }
    }
  }

  def mainLoop() = {
    assert(miniBatchSize <= data.size)

    assert(done == false)
    // Launch a thread to send the messages in the background
    broadcastThread.start()

    // Assume normalized loss and each machine scales gradient to size of the data.
    val lossScaleTerm = nData.toDouble / (miniBatchSize.toDouble * nData.toDouble)

    // Loop until done
    var t = 0
    while (!done) {
      grad *= 0.0 // Clear the gradient sum
      var b = 0
      while (b < miniBatchSize) {
        val ind = if (miniBatchSize < data.length) rnd.nextInt(data.length) else b
        lossFun.addGradient(primalVar, data(ind)._2, data(ind)._1, grad)
        b += 1
      }
      // Scale up the gradient
      grad *= lossScaleTerm

      // Add the regularizer to the gradient
      regularizer.addGradient(primalVar, params.regParam, grad)

      // Compute the new dual var:
      dualVar = (dualVar + dualSum) / (rcvdFrom.get().toDouble + 1.0) + grad

      // Set the learning rate
      eta_t = params.eta_0 / math.sqrt(t.toDouble + 1.0)

      // Apply the project using l2 prox operator
      primalVar = dualVar * (-eta_t)

      primalSum += primalVar

      t += 1
    }
    localIters = t
    primalVar = primalSum / t.toDouble

    broadcastThread.join()
  }

}


class DualAvgSGD extends BasicEmersonOptimizer with Serializable with Logging {
  var stats: Stats = null
  var totalTimeMs: Long = -1

  @transient var workers : RDD[DualAvgSGDWorker] = null


  override def initialize(params: EmersonParams,
                          lossFunction: LossFunction, regularizationFunction: Regularizer,
                          initialWeights: BV[Double],
                          rawData: RDD[Array[(Double, BV[Double])]]): Unit = {
    // Preprocess the data
    super.initialize(params, lossFunction, regularizationFunction, initialWeights, rawData)

    val primal0 = initialWeights.copy
    workers = data.mapPartitionsWithIndex { (ind, iter) =>
      if(DASetupBlock.initialized) {
        if (DASetupBlock.workers(ind) != null ) {
          Iterator(DASetupBlock.workers(ind))
        } else {
          throw new RuntimeException("Worker was evicted, dying lol!")
        }
      } else {

      val data: Array[(Double, BV[Double])] = iter.next()
      val workerName = UUID.randomUUID().toString
      val address = Worker.HACKakkaHost + workerName
      val hack = new DAWorkerCommunicationHack()
      logInfo(s"local address is $address")
      val aref = Worker.HACKworkerActorSystem.actorOf(Props(new DAWorkerCommunication(address, hack)), workerName)
      implicit val timeout = Timeout(15000 seconds)

      val f = aref ? new InternalMessages.WakeupMsg
      Await.result(f, timeout.duration).asInstanceOf[String]

        val worker = new DualAvgSGDWorker(subProblemId = ind,
          nSubProblems = nSubProblems, nData = nData.toInt, data = data,
          lossFun = lossFunction, params = params, regularizer = regularizationFunction,
          comm = hack.ref)
        worker.primalVar = primal0.copy
        DASetupBlock.workers(ind) = worker
        Iterator(worker)
      }
    }.cache()

    // collect the addresses
    val addresses = workers.map {
      if(DASetupBlock.initialized) {
        throw new RuntimeException("Worker was evicted, dying lol!")
      }
      w => w.comm.address
    }.collect()

    // Establish connections to all other workers
    workers.foreach { w =>
      DASetupBlock.initialized = true
      w.comm.connectToOthers(addresses)
    }

    // Ping Pong?  Just because?
    workers.foreach { w => w.comm.sendPingPongs() }


  }


  def statsMap(): Map[String, String] = {
    Map(
      "iterations" -> stats.avgLocalIters().x.toString,
      "iterInterval" -> stats.avgLocalIters().toString,
      "avgSGDIters" -> stats.avgSGDIters().toString,
      "avgMsgsSent" -> stats.avgMsgsSent().toString,
      "avgMsgsRcvd" -> stats.avgMsgsRcvd().toString,
      "primalAvgNorm" -> norm(stats.primalAvg(), 2).toString,
      "dualAvgNorm" -> norm(stats.dualAvg(), 2).toString,
      "consensusNorm" -> norm(finalW, 2).toString,
      "dualUpdates" -> stats.avgDualUpdates.toString,
      "runtime" -> totalTimeMs.toString,
      "stats" -> stats.toString
    )
  }

  var finalW: BV[Double] = null

  def optimize(): BV[Double] = {
    val startTimeNs = System.nanoTime()

    // Run all the workers
    workers.foreach( w => w.mainLoop() )
    // Collect the primal and dual averages
    stats =
      workers.map { w => w.getStats() }.reduce( _ + _ )

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    finalW = stats.primalAvg()
    finalW
  }
}

