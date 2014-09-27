package edu.berkeley.emerson

import java.util.UUID
import java.util.concurrent._
import java.util.concurrent.atomic._

import akka.actor._
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg.{norm, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.Logging
import org.apache.spark.deploy.worker.Worker
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

//
//case class AsyncSubProblem(data: Array[(Double, Vector)], comm: WorkerCommunication)

object HWInternalMessages {
  class WakeupMsg
  class PingPong
  class DeltaUpdate(val sender: Int,
                    val delta: BV[Double])
}

class HWWorkerCommunicationHack {
  var ref: HWWorkerCommunication = null
}

object HWSetupBlock {
  var initialized = false
  val workers = new Array[HOGWILDSGDWorker](128)

}


class HWWorkerCommunication(val address: String, val hack: HWWorkerCommunicationHack) extends Actor with Logging {
  hack.ref = this
  val others = new mutable.HashMap[Int, ActorSelection]
  var selfID: Int = -1

  @volatile var optimizer: HOGWILDSGDWorker = null

  def receive = {
    case ppm: InternalMessages.PingPong => {
      logInfo("new message from " + sender)
    }
    case m: InternalMessages.WakeupMsg => {
      logInfo("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: HWInternalMessages.DeltaUpdate => {
      if (optimizer != null) {
        optimizer.primalVar -= d.delta
        optimizer.msgsRcvd.getAndIncrement()
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

  def broadcastDeltaUpdate(delta: BV[Double]) {
    val msg = new HWInternalMessages.DeltaUpdate(selfID, delta)
    for (other <- others.values) {
      other ! msg
    }
  }
}



class HOGWILDSGDWorker(subProblemId: Int,
                       nSubProblems: Int,
                       nData: Int,
                       data: Array[(Double, BV[Double])],
                       lossFun: LossFunction,
                       params: EmersonParams,
                       regularizer: Regularizer,
                       val comm: HWWorkerCommunication)
  extends ADMMLocalOptimizer(subProblemId = subProblemId, nSubProblems,
			    nData = nData, data = data, lossFun = lossFun, regularizer, params)
  with Logging {

  comm.optimizer = this

  @volatile var done = false
  @volatile var msgsSent = 0
  @volatile var msgsRcvd = new AtomicInteger()
  @volatile var grad_delta: BV[Double] = BV.zeros(nDim)

  override def getStats() = {
    Stats(primalVar = primalVar, dualVar = dualVar,
      msgsSent = msgsSent, msgsRcvd = msgsRcvd.get(),
      localIters = localIters,
      dataSize = data.length)
  }

  val broadcastThread = new Thread {
    override def run {
      val startTime = System.currentTimeMillis()
      while (!done) {
        comm.broadcastDeltaUpdate(grad_delta.copy)
        msgsSent += 1
        grad_delta *= 0.0
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

      // Set the learning rate
      val eta_t = params.eta_0 / math.sqrt(t.toDouble + 1.0)

      // Scale the gradient by the learning rate
      grad *= eta_t

      // Take the gradient step
      primalVar -= grad

      // Accumulate the delta gradient to be sent to other machines
      grad_delta += grad

      t += 1
    }
    localIters = t
    primalVar
    broadcastThread.join()
  }

}


class HOGWILDSGD extends BasicEmersonOptimizer with Serializable with Logging {
  var stats: Stats = null
  var totalTimeMs: Long = -1

  @transient var workers : RDD[HOGWILDSGDWorker] = null


  override def initialize(params: EmersonParams,
                          lossFunction: LossFunction, regularizationFunction: Regularizer,
                          initialWeights: BV[Double],
                          rawData: RDD[Array[(Double, BV[Double])]]): Unit = {
    // Preprocess the data
    super.initialize(params, lossFunction, regularizationFunction, initialWeights, rawData)

    val primal0 = initialWeights.copy
    workers = data.mapPartitionsWithIndex { (ind, iter) =>
      if(HWSetupBlock.initialized) {
        if (HWSetupBlock.workers(ind) != null ) {
          Iterator(HWSetupBlock.workers(ind))
        } else {
          throw new RuntimeException("Worker was evicted, dying lol!")
        }
      } else {

      val data: Array[(Double, BV[Double])] = iter.next()
      val workerName = UUID.randomUUID().toString
      val address = Worker.HACKakkaHost + workerName
      val hack = new HWWorkerCommunicationHack()
      logInfo(s"local address is $address")
      val aref = Worker.HACKworkerActorSystem.actorOf(Props(new HWWorkerCommunication(address, hack)), workerName)
      implicit val timeout = Timeout(15000 seconds)

      val f = aref ? new InternalMessages.WakeupMsg
      Await.result(f, timeout.duration).asInstanceOf[String]

        val worker = new HOGWILDSGDWorker(subProblemId = ind,
          nSubProblems = nSubProblems, nData = nData.toInt, data = data,
          lossFun = lossFunction, params = params, regularizer = regularizationFunction,
          comm = hack.ref)
        worker.primalVar = primal0.copy
        HWSetupBlock.workers(ind) = worker
        Iterator(worker)
      }
    }.cache()

    // collect the addresses
    val addresses = workers.map {
      if(HWSetupBlock.initialized) {
        throw new RuntimeException("Worker was evicted, dying lol!")
      }
      w => w.comm.address
    }.collect()

    // Establish connections to all other workers
    workers.foreach { w =>
      HWSetupBlock.initialized = true
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
    // compute the final consensus value synchronously
    val nExamples = workers.map(w=>w.data.length).reduce(_+_)
    // Collect the primal and dual averages
    stats =
      workers.map { w => w.getStats() }.reduce( _ + _ )

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    finalW = stats.primalAvg()
    finalW
  }
}

