package edu.berkeley.emerson

import java.util.UUID
import java.util.concurrent._

import akka.actor._
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import edu.berkeley.emerson.InternalMessages.VectorUpdateMessage
import org.apache.spark.Logging
import org.apache.spark.deploy.worker.Worker
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
                            val primalVar: BV[Double], val dualVar: BV[Double])
}



class WorkerCommunication(val address: String, val hack: WorkerCommunicationHack) extends Actor with Logging {
  hack.ref = this
  val others = new mutable.HashMap[Int, ActorSelection]
  var selfID: Int = -1

  var inputQueue = new LinkedBlockingQueue[VectorUpdateMessage]()

  var worker: AsyncADMMWorker = null

  def receive = {
    case ppm: InternalMessages.PingPong => {
      logInfo("new message from " + sender)
    }
    case m: InternalMessages.WakeupMsg => {
      logInfo("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: InternalMessages.VectorUpdateMessage => {
      if(worker != null) {
        worker.receiveMsg(d.sender, d.primalVar, d.dualVar)
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
      // skip self
      if (!host.equals(address)) {
        //logInfo(s"Connecting to $host, $i")
        others.put(i, context.actorSelection(allHosts(i)))

        implicit val timeout = Timeout(15000 seconds)
        val f = others(i).resolveOne()
        Await.ready(f, Duration.Inf)
        logInfo(s"Connected to ${f.value.get.get}")
      } else {
        selfID = i
      }
      i += 1
    }
    if(selfID == -1) {
      println("SelfID is -1 !!!!!!!!!!!!!!")
      println(s"Address: $address")
      println(s"Hosts: ")
      allHosts.foreach(x => println(s"\t $x")) 
      throw new RuntimeException("Worker was evicted, dying not lol!")
    }
  }

  def sendPingPongs() {
    for (other <- others.values) {
      other ! new InternalMessages.PingPong
    }
  }

  def broadcastDeltaUpdate(primalVar: BV[Double], dualVar: BV[Double]) {
    val msg = new InternalMessages.VectorUpdateMessage(selfID, primalVar, dualVar)
    for (other <- others.values) {
      other ! msg
    }
  }
}


class AsyncADMMWorker(subProblemId: Int,
                      nSubProblems: Int,
                      nData: Int, 
                      data: Array[(Double, BV[Double])],
                      lossFun: LossFunction,
                      params: EmersonParams,
                      val regularizer: Regularizer,
                      val comm: WorkerCommunication)
    extends ADMMLocalOptimizer(subProblemId = subProblemId, nSubProblems = nSubProblems, nData = nData,
      data = data, lossFun = lossFun, params)
    with Logging {
  comm.worker = this

  @volatile var done = false
  @volatile var startTime = 0L
  @volatile var msgsSent = 0
  @volatile var msgsRcvd = 0
  @volatile var ranOnce = false

  override def getStats() = {
    Stats(primalVar = primalVar, dualVar = dualVar,
      msgsSent = msgsSent, msgsRcvd = msgsRcvd,
      localIters = localIters, sgdIters = sgdIters,
      dualUpdates = dualIters,
      dataSize = data.length)
  }


  val broadcastThread = new Thread {
    override def run {
      assert(!done)
      while (!done) {
        Thread.sleep(params.broadcastDelayMS)
        var primalSnapshot: BV[Double] = null
        var dualSnapshot: BV[Double] = null

        // Grab a snapshot and clear the shared variables
        data.synchronized {
          primalSnapshot = goodPrimal
          dualSnapshot = goodDual
          goodPrimal = null
          goodDual = null
        }

        if (primalSnapshot != null && dualSnapshot != null) {
          comm.broadcastDeltaUpdate(primalVar, dualVar)
        }

        msgsSent += 1
        // Check to see if we are done
        val elapsedTime = System.currentTimeMillis() - startTime
        done = elapsedTime > params.runtimeMS
      }
    }
  }
 
  @volatile var goodPrimal: BV[Double] = null
  @volatile var goodDual: BV[Double] = null

  var lastSend: Long = 0

  // PORKCHOP
  val solverLoopThread = new Thread {
    override def run {
      while (!done) {
        // val primalOld = primalVar.copy
        // val dualOld = dualVar.copy

        // Do a Dual update if the primal seems to be converging
        // dualUpdate(params.lagrangianRho / (localIters + 1.0).toDouble )
        dualUpdate(params.lagrangianRho)
        
        // Start over at primal consensus
        primalVar = primalConsensus.copy  // this was there before

        val timeRemainingMS = params.runtimeMS - (System.currentTimeMillis() - startTime)        
        // Run the primal update
        primalUpdate(timeRemainingMS)


        // data.synchronized {
        //   goodPrimal = primalCopy
        //   goodDual = dualCopy
        // }

        var currentTime = System.currentTimeMillis() 
  
        if (currentTime - lastSend > params.broadcastDelayMS) {
          val primalCopy = primalVar.copy
          val dualCopy = dualVar.copy
          comm.broadcastDeltaUpdate(primalCopy, dualCopy)
          lastSend = System.currentTimeMillis()
          currentTime = lastSend
        }

        receiveMsg(comm.selfID, primalVar, dualVar)
        primalConsensus = receivedPrimalConsensus
        System.out.println("primal consensus is "+primalConsensus.toString+ " vec is "+primalVar.toString)

        localIters += 1

        // Assess Termination
        val elapsedTime = currentTime - startTime
        done = elapsedTime >= params.runtimeMS

      }
      println(s"${comm.selfID}: Sent death message ${System.currentTimeMillis()}")
      // Kill the consumer thread
      val poisonMessage = new InternalMessages.VectorUpdateMessage(-2, null, null)
      comm.inputQueue.add(poisonMessage)
    }
  }

  val lastMsg = new Array[(BV[Double], BV[Double])](256)
  @volatile var primalSum = BV.zeros[Double](primalVar.size)
  @volatile var dualSum = BV.zeros[Double](dualVar.size)
  @volatile var receivedPrimalConsensus = BV.zeros[Double](primalConsensus.size)
  @volatile var sumTerms = 0

  def receiveMsg(srcId: Int, newPrimal: BV[Double], newDual: BV[Double]) {
    lastMsg.synchronized {
      if (lastMsg(srcId) != null) {
        val (oldPrimal, oldDual) = lastMsg(srcId)
        primalSum += (newPrimal - oldPrimal)
        dualSum += (newDual - oldDual)
        lastMsg(srcId) = (newPrimal, newDual)
      } else {
        sumTerms += 1
        lastMsg(srcId) = (newPrimal, newDual)
        primalSum += newPrimal
        dualSum += newDual
      }
      msgsRcvd += 1
      val primalAvg = primalSum/sumTerms.toDouble
      val dualAvg = dualSum/sumTerms.toDouble
      // Recompute the consensus variable
      receivedPrimalConsensus = regularizer.consensus(primalAvg, dualAvg,
					      nSolvers = sumTerms,
					      rho = rho, 
					      regParam = regParamScaled)
    }
  }

  def mainLoop() = {
    assert(!done)
    assert(!ranOnce)
    ranOnce = true
    startTime = System.currentTimeMillis()
    rho = params.rho0
    val primalOptimum =
      if (params.usePorkChop) {
        mainLoopAsync()
      } else {
        mainLoopSync()
      }
    primalOptimum
  }


  def mainLoopAsync() = {
    println(s"${comm.selfID}: Starting the main loop.")
    // Launch a thread to send the messages in the background
    solverLoopThread.start()
    //consumerThread.start()
    //broadcastThread.start()

    solverLoopThread.join()
    //consumerThread.join()
    //broadcastThread.join()

    println(s"${comm.selfID}: Finished main loop.")
    // Return the primal consensus value
    primalConsensus
  }

 // ASYNCADMM, not PORKCHOP
  def mainLoopSync() = {
    // Intialize global view of primalVars
    val allVars = new mutable.HashMap[Int, (BV[Double], BV[Double])]()
    var primalAvg = BV.zeros[Double](primalVar.size)
    var dualAvg = BV.zeros[Double](dualVar.size)

    // Loop until done
    while (!done) {
      // Do a dual update
      dualUpdate(params.lagrangianRho)

      // Run the primal update
      val timeRemainingMS = params.runtimeMS - (System.currentTimeMillis() - startTime)
      primalUpdate(timeRemainingMS)



      // Send the primal and dual
      comm.broadcastDeltaUpdate(primalVar, dualVar)
      msgsSent += 1

      // Collect latest variables from everyone
      var tiq = comm.inputQueue.poll()
      while (tiq != null) {
        allVars(tiq.sender) = (tiq.primalVar, tiq.dualVar)
        tiq = comm.inputQueue.poll()
        msgsRcvd += 1
      }
      allVars.put(comm.selfID, (primalVar, dualVar))

      // Compute primal and dual averages
      primalAvg *= 0.0
      dualAvg *= 0.0
      val msgIterator = allVars.values.iterator
      while (msgIterator.hasNext) {
        val (primal, dual) = msgIterator.next()
        primalAvg += primal
        dualAvg += dual
      }
      primalAvg /= allVars.size.toDouble
      dualAvg /= allVars.size.toDouble

      // Recompute the consensus variable
      primalConsensus = regularizer.consensus(primalAvg, dualAvg, 
					      nSolvers = allVars.size,
					      rho = rho, 
					      regParam = regParamScaled)

      // Reset the primal var
      primalVar = primalConsensus.copy

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
  val workers = new Array[AsyncADMMWorker](128)

}


class AsyncADMM extends BasicEmersonOptimizer with Serializable with Logging {

  var totalTimeMs: Long = -1

  var stats: Stats = null

  @transient var workers : RDD[AsyncADMMWorker] = null

  override def initialize(params: EmersonParams,
                 lossFunction: LossFunction, regularizationFunction: Regularizer,
                 initialWeights: BV[Double],
                 rawData: RDD[Array[(Double, BV[Double])]]): Unit = {
    // Preprocess the data
    super.initialize(params, lossFunction, regularizationFunction, initialWeights, rawData)

    val primal0 = initialWeights.copy
    workers = data.mapPartitionsWithIndex { (ind, iter) =>
      if(SetupBlock.initialized) {
        if (SetupBlock.workers(ind) != null ) {
          Iterator(SetupBlock.workers(ind))
        } else {
          throw new RuntimeException("Worker was evicted, dying lol!")
        }
      } else {
        val data: Array[(Double, BV[Double])] = iter.next()
        val workerName = UUID.randomUUID().toString
        val address = Worker.HACKakkaHost + workerName
        val hack = new WorkerCommunicationHack()
        logInfo(s"local address is $address")
        val aref = Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
        implicit val timeout = Timeout(15000 seconds)

        val f = aref ? new InternalMessages.WakeupMsg
        Await.result(f, timeout.duration).asInstanceOf[String]

        val worker = new AsyncADMMWorker(subProblemId = ind,
          nSubProblems = nSubProblems, nData = nData.toInt , data = data,
          lossFun = lossFunction, params = params, regularizer = regularizationFunction,
          comm = hack.ref)
        worker.primalVar = primal0.copy
        worker.dualVar = primal0.copy
        SetupBlock.workers(ind) = worker
        Iterator(worker)
      }
    }.cache()

    // collect the addresses
    val addresses = workers.map {
      w => w.comm.address
    }.collect()

    // Establish connections to all other workers
    workers.foreach { w =>
      SetupBlock.initialized = true
      w.comm.connectToOthers(addresses)
    }

    addresses.foreach( a => println(s"\t $a") )
    println(s"Num Addresses: ${addresses.length}")

    // Ping Pong?  Just because?
    workers.foreach {
      w => w.comm.sendPingPongs()
    }

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
    stats = workers.map{
      w => w.mainLoop()
      w.getStats()
    }.reduce( _ + _ )
    
    val regParamScaled = params.regParam // * params.admmRegFactor
    finalW = regularizationFunction.consensus(
               stats.primalAvg, stats.dualAvg,
				       stats.nWorkers, 
				       params.rho0,
				       regParam = regParamScaled)
    println(stats.primalAvg())
    println(stats.dualAvg())
    println(finalW)
    println(s"Final norm: ${norm(finalW, 2)}")


    val totalTimeNs = System.nanoTime() - startTimeNs
    totalTimeMs = TimeUnit.MILLISECONDS.convert(totalTimeNs, TimeUnit.NANOSECONDS)

    finalW
  }
}

