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
import java.util
import akka.routing.BroadcastRouter
import com.twitter.chill.{ScalaKryoInstantiator, KryoPool}

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
  class PackedVectorUpdateMessage(val bytes: Array[Byte])

}

class WorkerCommunication(val address: String, val hack: WorkerCommunicationHack) extends Actor with Logging {
  hack.ref = this
  val others = new mutable.HashMap[Int, ActorRef]
  var selfID: Int = -1

  var inputQueue = new LinkedBlockingQueue[VectorUpdateMessage]()

  val kryoPool = KryoPool.withByteArrayOutputStream(50, new ScalaKryoInstantiator())

  def receive = {
    case ppm: InternalMessages.PingPong => {
      logInfo("new message from " + sender)
    }
    case m: InternalMessages.WakeupMsg => {
      logInfo("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: InternalMessages.PackedVectorUpdateMessage => {
      inputQueue.add(kryoPool.fromBytes(d.bytes, classOf[InternalMessages.VectorUpdateMessage]))
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
        val selection = context.actorSelection(allHosts(i))

        implicit val timeout = Timeout(150000 seconds)
        val f = selection.resolveOne()
        Await.ready(f, Duration.Inf)
        val ref = f.value.get.get
        others.put(i, ref)

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
    val msg = new InternalMessages.PackedVectorUpdateMessage(kryoPool.toBytesWithoutClass(new InternalMessages.VectorUpdateMessage(selfID, primalVar, dualVar, nExamples)))
    for(other <- others.values) {
      other ! msg
    }
  }
}


class AsyncADMMWorker(subProblemId: Int,
                      data: Array[(Double, BV[Double])],
                      gradient: FastGradient,
                      params: ADMMParams,
                      val consensus: ConsensusFunction,
                      val comm: WorkerCommunication,
                      val nSubProblems: Int)
    extends SGDLocalOptimizer(subProblemId = subProblemId, data = data, gradient = gradient, params)
    with Logging {

  @volatile var done = false
  @volatile var startTime = 0L
  @volatile var msgsSent = 0
  @volatile var ranOnce = false

  override def getStats() = {
    WorkerStats(primalVar = primalVar, dualVar = dualVar,
      msgsSent = msgsSent, localIters = localIters, sgdIters = sgdIters,
      dataSize = data.length)
  }

  val broadcastThread = new Thread {
    override def run {
      while (!done) {
        comm.broadcastDeltaUpdate(primalVar, dualVar, data.length)
        msgsSent += 1
        Thread.sleep(params.broadcastDelayMS)
        // Check to see if we are done
        val elapsedTime = System.currentTimeMillis() - startTime
        done = elapsedTime > params.runtimeMS
      }
    }
  }


  val solverLoopThread = new Thread {
    override def run {
      while (!done) {
        val primalOld = primalVar.copy
        val timeRemainingMS = params.runtimeMS - (System.currentTimeMillis() - startTime)
        // Run the primal update
        primalUpdate(timeRemainingMS)
        // Do a Dual update if the primal seems to be converging
        if (norm(primalOld - primalVar, 2) < 0.01) {
          dualUpdate(params.lagrangianRho)
        }
        localIters += 1
      }
      // Kill the consumer thread
      val poisonMessage = new InternalMessages.VectorUpdateMessage(-1, null, null, -1)
      comm.inputQueue.add(poisonMessage)
    }
  }

  val consumerThread = new Thread {
    override def run {
      // Intialize global view of primalVars
      val allVars = new mutable.HashMap[Int, (BV[Double], BV[Double], Int)]()
      while (!done) {
        // Collect latest variables from everyone
        allVars.put(comm.selfID, (primalVar, dualVar, data.length))
        var tiq = comm.inputQueue.take()
        while (tiq != null) {
          if (tiq.nExamples == -1) {
            done = true
          } else {
            allVars(tiq.sender) = (tiq.primalVar, tiq.dualVar, tiq.nExamples)
            tiq = comm.inputQueue.poll()
          }
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
        primalConsensus = consensus(primalAvg, dualAvg, nSubProblems, rho, params.regParam)

        // If the consensus changed then update the dual once (why not?)
        if (norm(primalConsensus - primalConsensusOld, 2) > 0.01) {
          dualUpdate(params.lagrangianRho)
        }
      }
    }
  }

  def mainLoop() = {
    assert(!done)
    assert(!ranOnce)
    ranOnce = true
    startTime = System.currentTimeMillis()
    rho = params.rho0
    broadcastThread.start()
    val primalOptimum = if (params.usePorkChop) {
      mainLoopAsync()
    } else {
      mainLoopSync()
    }
    broadcastThread.join()
    primalOptimum
  }


  def mainLoopAsync() = {
    // Launch a thread to send the messages in the background
    solverLoopThread.start()
    consumerThread.start()
    solverLoopThread.join()
    consumerThread.join()
    // Return the primal consensus value
    primalConsensus
  }


  def mainLoopSync() = {
    // Intialize global view of primalVars
    val allVars = new mutable.HashMap[Int, (BV[Double], BV[Double], Int)]()
    // Loop until done
    while (!done) {
      // Reset the primal var
      // primalVar = primalConsensus.copy

      // Run the primal update
      val timeRemainingMS = params.runtimeMS - (System.currentTimeMillis() - startTime)
      primalUpdate(timeRemainingMS)


      // Collect latest variables from everyone
      allVars.put(comm.selfID, (primalVar, dualVar, data.length))
      var tiq = comm.inputQueue.poll()
      val receivedMsgs = tiq != null
      while (tiq != null) {
        allVars(tiq.sender) = (tiq.primalVar, tiq.dualVar, tiq.nExamples)
        tiq = comm.inputQueue.poll()
      }

      // Compute primal and dual averages
      var (primalAvg, dualAvg, nTotalExamples) = allVars.values.iterator.map {
        case (primal, dual, nExamples) =>
          (primal * nExamples.toDouble, dual * nExamples.toDouble, nExamples)
      }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
      primalAvg /= nTotalExamples.toDouble
      dualAvg /= nTotalExamples.toDouble

      // Recompute the consensus variable
      val primalConsensusOld = primalConsensus.copy
      primalConsensus = consensus(primalAvg, dualAvg, nSubProblems, rho, params.regParam)


      if (params.adaptiveRho) {
        // Compute the residuals
        val primalResidual = allVars.values.iterator.map {
          case (primalVar, dualVar, nExamples) => norm(primalVar - primalConsensus, 2) * nExamples
        }.sum / nTotalExamples.toDouble
        val dualResidual = rho * norm(primalConsensus - primalConsensusOld, 2)
        // Rho update from Boyd text
        if (rho == 0.0) {
           rho = 1.0
        } else if (primalResidual > 10.0 * dualResidual && rho < 8.0) {
           rho = 2.0 * rho
           println(s"Increasing rho: $rho")
        } else if (dualResidual > 10.0 * primalResidual && rho > 0.01) {
           rho = rho / 2.0
           println(s"Decreasing rho: $rho")
        }
        dualUpdate(rho)
      } else {
        dualUpdate(params.lagrangianRho)
      }

      // Check to see if we are done
      val elapsedTime = System.currentTimeMillis() - startTime
      done = elapsedTime > params.runtimeMS
      localIters += 1
    }

    // Return the primal consensus value
    primalConsensus
  }

}

object SetupBlock {
  var initialized = false
}


class AsyncADMM(val params: ADMMParams, val gradient: FastGradient, var consensus: ConsensusFunction)
  extends Optimizer with Serializable with Logging {

  var totalTimeMs: Long = -1

  var stats: WorkerStats = null

  @transient var workers : RDD[AsyncADMMWorker] = null

  def setup(input: RDD[(Double, Vector)], primal0: BV[Double]) {
    val nSubProblems = input.partitions.length

    workers = input.mapPartitionsWithIndex { (ind, iter) =>
      val data: Array[(Double, BV[Double])] =
        iter.map { case (label, features) => (label, features.toBreeze) }.toArray
      val workerName = UUID.randomUUID().toString
      val address = Worker.HACKakkaHost+workerName
      val hack = new WorkerCommunicationHack()
      logInfo(s"local address is $address")
      val aref = Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
      implicit val timeout = Timeout(15000 seconds)

      val f = aref ? new InternalMessages.WakeupMsg
      Await.result(f, timeout.duration).asInstanceOf[String]

      val worker = new AsyncADMMWorker(subProblemId = ind, nSubProblems = nSubProblems, data = data,
        gradient = gradient, params = params, consensus = consensus, comm = hack.ref)
      worker.primalVar = primal0.copy
      worker.dualVar = primal0.copy
      Iterator(worker)
     }.cache()

    // collect the addresses
    val addresses = workers.map {
      if(SetupBlock.initialized) {
        throw new RuntimeException("Worker was evicted, dying lol!")
      }
      w => w.comm.address
    }.collect()

    // Establish connections to all other workers
    workers.foreach { w =>
      SetupBlock.initialized = true
      w.comm.connectToOthers(addresses)
    }

    // Ping Pong?  Just because?
    workers.foreach { w => w.comm.sendPingPongs() }

    input.unpersist(true)
    workers.foreach( f => System.gc() )
  }

  def optimize(input: RDD[(Double, Vector)], primal0: Vector): Vector = {
    // Initialize the cluster
    setup(input, primal0.toBreeze)

    val startTimeNs = System.nanoTime()

    // Run all the workers
    stats = workers.map{
      w => w.mainLoop()
      w.getStats()
    }.reduce( _ + _ )
    val finalW = consensus(stats.primalAvg, stats.dualAvg, stats.nWorkers, params.rho0,
      params.regParam)

    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    Vectors.fromBreeze(finalW)
  }
}

