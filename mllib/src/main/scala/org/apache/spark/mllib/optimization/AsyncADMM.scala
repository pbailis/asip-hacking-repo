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

case class AsyncSubProblem(data: Array[(Double, Vector)], comm: WorkerCommunication)

// fuck actors
class WorkerCommunicationHack {
  var ref: WorkerCommunication = null
}

object InternalMessages {
  class WakeupMsg
  class PingPong
  class VectorUpdateMessage(val delta: BV[Double], val sender: Int)
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

  def broadcastDeltaUpdate(delta: BV[Double]) {
    val msg = new InternalMessages.VectorUpdateMessage(delta, selfID)
    for (other <- others.values) {
      other ! msg
    }
  }
}

class DeltaBroadcaster(val comm: WorkerCommunication,
                        var w: BV[Double],
                        val scheduler: ScheduledExecutorService,
                        @volatile var broadcastsRemaining: Int,
                        var broadcastPeriodMs: Long) extends Runnable with Logging {
  var prev_w = w
  @volatile var done = false

  @Override def run = {
    logInfo(s"Broadcasting at a rate of $broadcastPeriodMs; $broadcastsRemaining remaining")
    comm.broadcastDeltaUpdate(w)

    broadcastsRemaining -= 1
    if(broadcastsRemaining > 0) {
      scheduleThis()
    } else {
      done = true
    }
  }

  def scheduleThis() {
    scheduler.schedule(this, broadcastPeriodMs, TimeUnit.MILLISECONDS)
  }
}

class AsyncSGDLocalOptimizer(val gradient: Gradient,
                             val updater: Updater,
                             val totalSeconds: Double,
                             val numberOfParamBroadcasts: Int,
                             val miniBatchFraction: Double) extends Logging {
  var eta_0: Double = 1.0
  var maxIterations: Int = Integer.MAX_VALUE
  var epsilon: Double = 0.001

  val scheduler = Executors.newScheduledThreadPool(1)

  def apply(subproblem: AsyncSubProblem,
            w0: BV[Double], init_w_consensus: BV[Double], dualVar: BV[Double],
            rho0: Double, regParam: Double): BV[Double] = {

    val lastHeardVectors = new mutable.HashMap[Int, BV[Double]]()
    var w_consensus = init_w_consensus
    var rho = rho0

    for(i <- subproblem.comm.others.keySet) {
      lastHeardVectors.put(i, w0)
    }

    val db = new DeltaBroadcaster(subproblem.comm,
                                  w0,
                                  scheduler,
                                  numberOfParamBroadcasts,
      (totalSeconds*1000.0/numberOfParamBroadcasts).toLong)

    db.scheduleThis()

    val transverseIteratorQueue = subproblem.comm.inputQueue

    val nExamples = subproblem.data.length
    val dim = subproblem.data(0)._2.size
    val miniBatchSize = math.max(1, math.min(nExamples, (miniBatchFraction * nExamples).toInt))
    val rnd = new java.util.Random()
    var residual = 1.0
    var w = w0.copy
    var grad = BV.zeros[Double](dim)
    var t = 0
    while(!db.done) {

      residual = 1.0
      while(residual > 0.001) {
        grad *= 0.0 // Clear the gradient sum
        var b = 0
        while (b < miniBatchSize) {
          val ind = if (miniBatchSize < nExamples) rnd.nextInt(nExamples) else b
          val (label, features) = subproblem.data(ind)
          gradient.compute(features, label, Vectors.fromBreeze(w), Vectors.fromBreeze(grad))
          b += 1
        }
        // Normalize the gradient to the batch size
        grad /= miniBatchSize.toDouble
        // Add the lagrangian + augmenting term.
        grad += dualVar + (w - w_consensus) * rho
        // axpy(rho, dualVar + w - w_consensus, grad)
        // Set the learning rate
        val eta_t = eta_0 / (t.toDouble + 1.0)
        // w = w + eta_t * point_gradient
        // axpy(-eta_t, grad, w)
        val wOld = w.copy
        w = updater.compute(Vectors.fromBreeze(w), Vectors.fromBreeze(grad), eta_t, 1, regParam)._1.toBreeze
        // Compute residual.  This is a decaying residual definition.
        // residual = (eta_t * norm(grad, 2.0) + residual) / 2.0
        residual = norm(w - wOld, 2.0)
        t += 1
      }

      // process any inbound deltas that arrived during the last minibatch
      var changed = false
      var tiq = transverseIteratorQueue.poll()
      while(tiq != null) {
        lastHeardVectors(tiq.sender) = tiq.delta
        tiq = transverseIteratorQueue.poll()
        changed = true
      }


      if(changed) {
        lastHeardVectors(subproblem.comm.selfID) = w

        val w_consensus_old = w_consensus
        w_consensus = lastHeardVectors.values.reduce(_ + _) / lastHeardVectors.size.toDouble

//        if (norm(w_consensus_old - w_consensus, 2) < 0.01) {
//          dualVar += (w - w_consensus)
//        }

        dualVar += (w - w_consensus) * (rho / numberOfParamBroadcasts.toDouble)

        val dualResidual = rho * norm(w_consensus - w_consensus_old, 2)
        val primalResidual = (lastHeardVectors.values.iterator.map(x => norm(x - w_consensus,2)).sum) /
          lastHeardVectors.size.toDouble
        // Rho upate from Boyd text
        if (rho == 0.0) {
          rho = 1.0
        } else if (primalResidual > 10.0 * dualResidual) {
          rho = 2.0 * rho
          println(s"Increasing rho: $rho")
        } else if (dualResidual > 10.0 * primalResidual) {
          rho = rho / 2.0
          println(s"Decreasing rho: $rho")
        }


        val propCorrectConsensu =
              subproblem.data.iterator.map { case (y,x) => if (x.toBreeze.dot(w_consensus) * (y * 2.0 - 1.0) > 0.0) 1 else 0 }
              .reduce(_ + _).toDouble / nExamples.toDouble
        val stats = s"STATS: t = $t, primalResid = $primalResidual, dualResidual = $dualResidual,  accuracy = $propCorrectConsensu"
        println(s"$rho  $stats")
        //println(s"(w_consesnsus, w_local): $stats,\n \t ${w_consensus},   ${w},    ${dualVar}")
//        println("----------")
//        lastHeardVectors.values.foreach(v => println(s"\t $v"))
      }

      t = 0

      // make sure our outbound sender is up to date!
      db.w = w.copy
    }

    w
  }
}

class AsyncADMMwithSGD(val gradient: Gradient, val updater: Updater)  extends Optimizer with Logging {
  var localSubProblems: RDD[AsyncSubProblem] = null
  var totalSeconds = 10
  var numberOfParamBroadcasts = 10
  var miniBatchFraction: Double = 0.1

  var regParam = 0.001

  def setup(input: RDD[(Double, Vector)]) = {
    localSubProblems = input.mapPartitions {
      iter =>
        val workerName = UUID.randomUUID().toString
        val address = Worker.HACKakkaHost+workerName
        val hack = new WorkerCommunicationHack()
        logInfo(s"local address is $address")
        val aref= Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
        implicit val timeout = Timeout(15 seconds)

        val f = aref ? new InternalMessages.WakeupMsg
        Await.result(f, timeout.duration).asInstanceOf[String]

        Iterator(new AsyncSubProblem(iter.toArray, hack.ref))
    }.cache()

    val addresses = localSubProblems.map { w => w.comm.address }.collect()

    localSubProblems.foreach {
      w => w.comm.connectToOthers(addresses)
    }

    localSubProblems.foreach {
      w => w.comm.sendPingPongs()
    }
  }

  override def optimize(rawData: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    // TODO: run setup code outside of this loop
    val subProblems = setup(rawData)

    var rho  = 1.0
    val dim = rawData.map(block => block._2.size).first()
    var w_0 = BV.zeros[Double](dim)
    var w_avg = w_0.copy
    var dual_var = w_0.copy

    // TODO: average properly
    val avg = localSubProblems.map {
      p =>
        val solver = new AsyncSGDLocalOptimizer(gradient, updater, totalSeconds, numberOfParamBroadcasts, miniBatchFraction)
        solver(p, w_0, w_avg, dual_var, rho, regParam)
    }.reduce(_ + _)/localSubProblems.count().toDouble

    Vectors.fromBreeze(avg)
  }
}
