package org.apache.spark.mllib.optimization

import java.util.UUID
import java.util.concurrent._

import akka.actor._
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.Logging
import org.apache.spark.deploy.worker.Worker
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.InternalMessages.VectorUpdateMessage
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

//
//case class AsyncSubProblem(data: Array[(Double, Vector)], comm: WorkerCommunication)

// fuck actors
class WorkerCommunicationHack {
  var ref: WorkerCommunication = null
}

object InternalMessages {
  class WakeupMsg
  class PingPong
  class VectorUpdateMessage(val sender: Int,
                            val primalVar: BV[Double], val dualVar: BV[Double], val nExamples: Int)
}

class WorkerCommunication(val address: String, val hack: WorkerCommunicationHack) extends Actor with Logging {
  hack.ref = this
  val others = new mutable.HashMap[Int, ActorSelection]
  var selfID: Int = -1

  var inputQueue = new LinkedBlockingQueue[VectorUpdateMessage]()

  def receive = {
    case ppm: InternalMessages.PingPong => {
      logInfo("new message from " + sender)
    }
    case m: InternalMessages.WakeupMsg => {
      logInfo("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: InternalMessages.VectorUpdateMessage => {
      inputQueue.add(d)
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

  def broadcastDeltaUpdate(primalVar: BV[Double], dualVar: BV[Double], nExamples: Int) {
    val msg = new InternalMessages.VectorUpdateMessage(selfID, primalVar, dualVar, nExamples)
    for (other <- others.values) {
      other ! msg
    }
  }
}



class AsyncADMMWorker(subProblemId: Int,
                      val nSubProblems: Int,
                      data: Array[(Double, BV[Double])],
                      primalVar0: BV[Double],
                      gradient: FastGradient,
                      val consensus: ConsensusFunction,
                      val regParam: Double,
                      eta_0: Double,
                      epsilon: Double,
                      maxIterations: Int,
                      miniBatchSize: Int,
                      var rho: Double,
                      val comm: WorkerCommunication,
                      val broadcastDelayMS: Int)
  extends SGDLocalOptimizer(subProblemId = subProblemId, data = data, primalVar = primalVar0.copy,
    gradient = gradient, eta_0 = eta_0, epsilon = epsilon, maxIterations = maxIterations,
    miniBatchSize = miniBatchSize)
  with Logging {


  @volatile var done = false

  var primalConsensus = primalVar0.copy

  var commStages = 0
  val broadcastThread = new Thread {
    override def run {
      while (!done) {
        comm.broadcastDeltaUpdate(primalVar, dualVar, data.length)
        commStages += 1
        Thread.sleep(broadcastDelayMS)
      }
    }
  }


  def mainLoop(runTimeMS: Int = 1000) = {
    done = false
    // Launch a thread to send the messages in the background
    broadcastThread.start()

    // Intialize global view of primalVars
    val allVars = new mutable.HashMap[Int, (BV[Double], BV[Double], Int)]()

    var loopIter = 0
    val startTime = System.currentTimeMillis()
    // Loop until done
    while (!done) {
      // Reset the primal var
      primalVar = primalConsensus.copy
      // Run the primal update
      primalUpdate(primalConsensus, rho)

      // Collect latest variables from everyone
      allVars.put(comm.selfID, (primalVar, dualVar, data.length))
      var tiq = comm.inputQueue.poll()
      val receivedMsgs = tiq != null
      while (tiq != null) {
        allVars(tiq.sender) = (tiq.primalVar, tiq.dualVar, tiq.nExamples)
        tiq = comm.inputQueue.poll()
      }

      // Compute primal and dual averages
      var (primalAvg, dualAvg) = allVars.values.iterator.map {
        case (primal, dual, nExamples) => (primal * nExamples.toDouble, dual * nExamples.toDouble)
      }.reduce((a, b) => (a._1 + b._1, a._2 + b._2))
      val nTotalExamples = allVars.values.iterator.map {
        case (primal, dual, nExamples) => nExamples
      }.sum
      primalAvg /= nTotalExamples.toDouble
      dualAvg /= nTotalExamples.toDouble

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      primalConsensus = consensus(primalAvg, dualAvg, nSubProblems, rho, regParam)

      // Compute the residuals
      //val primalResidual = allVars.values.iterator.map {
      //  case (primalVar, dualVar, nExamples) => norm(primalVar - primalConsensus, 2) * nExamples
      //}.sum / nTotalExamples.toDouble
      //val dualResidual = rho * norm(primalConsensus - primalConsensusOld, 2)

      // // Rho update from Boyd text
      // if (rho == 0.0) {
      //   rho = 1.0
      // } else if (primalResidual > 10.0 * dualResidual && rho < 8.0) {
      //   rho = 2.0 * rho
      //   println(s"Increasing rho: $rho")
      // } else if (dualResidual > 10.0 * primalResidual && rho > 0.1) {
      //   rho = rho / 2.0
      //   println(s"Decreasing rho: $rho")
      // }


      //if(loopIter % 10 == 0) {
        // Run the dual update
        dualUpdate(primalConsensus, rho)
      //}
      //loopIter += 1

      // Check to see if we are done
      val elapsedTime = System.currentTimeMillis() - startTime
      done = elapsedTime > runTimeMS
    }
    // Run the primal update
    ///primalUpdate(primalConsensus, rho)

    // Return the primal consensus value
    primalConsensus
  }

}


class AsyncADMMwithSGD(val gradient: FastGradient, var consensus: ConsensusFunction) extends Optimizer with Serializable with Logging {

  var runtimeMS: Int = 5000
  var paramBroadcastPeriodMs = 100
  var regParam: Double = 1.0
  var epsilon: Double = 1.0e-5
  var eta_0: Double = 1.0
  var localEpsilon: Double = 0.001
  var localMaxIterations: Int = Integer.MAX_VALUE
  var miniBatchSize: Int = 10
  var displayLocalStats: Boolean = true
  var broadcastDelayMS: Int = 100
  var commStages: Int = 0
  var rho: Double = 1.0

  @transient var workers : RDD[AsyncADMMWorker] = null

  def setup(input: RDD[(Double, Vector)], primal0: BV[Double]) {
    val nSubProblems = input.partitions.length

    workers = input.mapPartitionsWithIndex { (ind, iter) =>
      val data: Array[(Double, BV[Double])] =
        iter.map { case (label, features) => (label, features.toBreeze)}.toArray
      val workerName = UUID.randomUUID().toString
      val address = Worker.HACKakkaHost+workerName
      val hack = new WorkerCommunicationHack()
      logInfo(s"local address is $address")
      val aref = Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
      implicit val timeout = Timeout(15 seconds)

      val f = aref ? new InternalMessages.WakeupMsg
      Await.result(f, timeout.duration).asInstanceOf[String]

      val worker = new AsyncADMMWorker(subProblemId = ind, nSubProblems = nSubProblems, data = data,
        primalVar0 = primal0.copy, gradient = gradient, consensus = consensus, regParam = regParam,
        eta_0 = eta_0, epsilon = localEpsilon, maxIterations = localMaxIterations,
        miniBatchSize = miniBatchSize, rho = rho, comm = hack.ref, broadcastDelayMS = broadcastDelayMS)

      Iterator(worker)
     }.cache()

    // collect the addresses
    val addresses = workers.map { w => w.comm.address }.collect()

    // Establish connections to all other workers
    workers.foreach { w =>
      w.comm.connectToOthers(addresses)
    }

    // Ping Pong?  Just because?
    workers.foreach { w => w.comm.sendPingPongs() }
  }

  var totalTimeMs: Long = -1

  def optimize(input: RDD[(Double, Vector)], primal0: Vector): Vector = {
    // Initialize the cluster
    setup(input, primal0.toBreeze)

    val startTimeNs = System.nanoTime()

    // Run all the workers
    var (primalAvg, dualAvg, nExamples, _commStages) = workers.map{
      w => w.mainLoop(runtimeMS)
      (w.primalVar * w.data.length.toDouble, w.dualVar * w.data.length.toDouble, w.data.length, w.commStages)
    }.reduce( (a,b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4) )
    // compute the final consensus value synchronously
    primalAvg /= nExamples.toDouble
    dualAvg /= nExamples.toDouble
    val rhoFinal = rho
    val finalW = consensus(primalAvg, dualAvg, workers.partitions.length, rhoFinal, regParam)

    commStages = _commStages

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    Vectors.fromBreeze(finalW)
  }
}

