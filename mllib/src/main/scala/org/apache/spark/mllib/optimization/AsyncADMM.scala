package org.apache.spark.mllib.optimization

import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import akka.actor._
import org.apache.spark.Logging
import org.apache.spark.deploy.worker.Worker
import scala.language.postfixOps
import java.util.UUID
import java.util

import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.duration._
import scala.concurrent.Await
import scala.collection.mutable
import org.apache.spark.mllib.linalg.Vector
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, _}
import org.apache.spark.mllib.optimization.InternalMessages.VectorUpdateMessage
import java.util.concurrent.{ScheduledExecutorService, ScheduledThreadPoolExecutor, TimeUnit, Executors}

case class AsyncSubProblem(data: Array[(Double, Vector)], comm: WorkerCommunication)

// fuck actors
class WorkerCommunicationHack {
  var ref: WorkerCommunication = null
}

object InternalMessages {
  class WakeupMsg
  class PingPong
  class VectorUpdateMessage(val delta: BV[Double])
}

class WorkerCommunication(val address: String, val hack: WorkerCommunicationHack) extends Actor {
  hack.ref = this
  var others = new mutable.HashMap[Int, ActorSelection]

  var inputQueue: java.util.Vector[BV[Double]] = new util.Vector[BV[Double]]()

  def receive = {
    case ppm: InternalMessages.PingPong => {
      println("new message from " + sender)
      sender ! "gotit!"
    }
    case m: InternalMessages.WakeupMsg => {
      println("activated local!"); sender ! "yo"
    }
    case s: String => println(s)
    case d: InternalMessages.VectorUpdateMessage => inputQueue.add(d.delta)
    case _ => println("hello, world!")
  }

  def shuttingDown: Receive = {
    case _ => println("GOT SHUTDOWN!")
  }

  def connectToOthers(allHosts: Array[String]) {
    var i = 0
    for (host <- allHosts) {
      if (!host.equals(address)) {
        others.put(i, context.actorSelection(allHosts(i)))
        /*
        others(i) ! new PingPong
        implicit val timeout = Timeout(15 seconds)

        val f = others(i).resolveOne()
        Await.ready(f, Duration.Inf)
        println(f.value.get.get)
        */
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
    val msg = new InternalMessages.VectorUpdateMessage(delta)
    for (other <- others.values) {
      other ! msg
    }
  }
}

class DeltaBroadcaster(val comm: WorkerCommunication,
                        var w: BV[Double],
                        val scheduler: ScheduledExecutorService,
                        var broadcastsRemaining: Int,
                        var broadcastPeriodMs: Long) extends Runnable {
  var prev_w = w
  @volatile var done = false

  @Override def run = {
    comm.broadcastDeltaUpdate(prev_w - w)
    prev_w = w

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
                             val numberOfParamBroadcasts: Int) extends Logging {
  var eta_0: Double = 1.0
  var maxIterations: Int = Integer.MAX_VALUE
  var epsilon: Double = 0.001
  var miniBatchFraction: Double = 0.1

  val scheduler = Executors.newScheduledThreadPool(1)

  def apply(subproblem: AsyncSubProblem,
            w0: BV[Double], w_consensus: BV[Double], dualVar: BV[Double],
            rho: Double, regParam: Double): BV[Double] = {


    val db = new DeltaBroadcaster(subproblem.comm,
                                  w0,
                                  scheduler,
                                  numberOfParamBroadcasts,
      (totalSeconds*1000/numberOfParamBroadcasts).toLong)

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
      // grad += rho * (dualVar + w - w_avg)
      axpy(rho, dualVar + w - w_consensus, grad)
      // Set the learning rate
      val eta_t = eta_0 / (t.toDouble + 1.0)
      // w = w + eta_t * point_gradient
      // axpy(-eta_t, grad, w)
      val wOld = w.copy
      w = updater.compute(Vectors.fromBreeze(w), Vectors.fromBreeze(grad), eta_t, 1, regParam)._1.toBreeze
      // Compute residual.  This is a decaying residual definition.
      // residual = (eta_t * norm(grad, 2.0) + residual) / 2.0
      residual = norm(w - wOld, 2.0)
      if((t % 10) == 0) {
        logInfo(s"Residual: $residual ")
      }
      t += 1

      // process any inbound deltas that arrived during the last minibatch
      if(!transverseIteratorQueue.isEmpty) {
        val tiq_it = transverseIteratorQueue.iterator()
        while(tiq_it.hasNext) {
          val delta = tiq_it.next()
          w_consensus += delta
        }
      }

      // make sure our outbound sender is up to date!
      db.w = w
    }

    w
  }
}

class AsyncADMMwithSGD(val gradient: Gradient, val updater: Updater)  extends Optimizer with Logging {
  var localSubProblems: RDD[AsyncSubProblem] = null
  var totalSeconds = 10
  var numberOfParamBroadcasts = 10

  var regParam = 1.0

  def setup(input: RDD[(Double, Vector)]) = {
    localSubProblems = input.mapPartitions {
      iter =>
        val workerName = UUID.randomUUID().toString
        val address = Worker.HACKakkaHost+workerName
        val hack = new WorkerCommunicationHack()
        println(address)
        val aref= Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
        implicit val timeout = Timeout(15 seconds)

        val f = aref ? new InternalMessages.WakeupMsg
        Await.result(f, timeout.duration).asInstanceOf[String]

        Iterator(new AsyncSubProblem(iter.toArray, hack.ref))
    }

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

    var rho  = 0.0
    val dim = rawData.map(block => block._2.size).first()
    var w_0 = BV.zeros[Double](dim)
    var w_avg = w_0.copy
    var dual_var = w_0.copy

    // TODO: average properly
    val avg = localSubProblems.map {
      p =>
        val solver = new AsyncSGDLocalOptimizer(gradient, updater, totalSeconds, numberOfParamBroadcasts)
        solver.apply(p, w_0, w_avg, dual_var, rho, regParam)
    }.reduce(_ + _)/localSubProblems.count().toDouble

    Vectors.fromBreeze(avg)
  }
}
