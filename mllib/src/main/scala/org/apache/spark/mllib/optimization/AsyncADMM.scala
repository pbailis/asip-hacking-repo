package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import breeze.linalg._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm}
import breeze.util.DoubleImplicits
import akka.actor.{Props, ActorSelection, Actor}
import scala.collection.mutable
import org.apache.spark.rdd.RDD
import java.util.UUID
import org.apache.spark.deploy.worker.Worker
import akka.util.Timeout
import scala.concurrent.Await

import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.duration._
import scala.concurrent.Await
import scala.collection.mutable
import scala.util.Random
import org.apache.spark.Logging

import scala.language.postfixOps
import java.util.concurrent.Semaphore


// fuck actors
class WorkerCommunicationHack {
  var ref: WorkerCommunication = null
}

object InternalMessages {
  class WakeupMsg
  class PingPong
  case class CurrentVector(v: BV[Double])
}

class WorkerCommunication(val address: String, val hack: WorkerCommunicationHack) extends Actor {
  hack.ref = this
  var others = new mutable.HashMap[Int, ActorSelection]

  var currentAvg: BV[Double] = null

  val newDataSemaphore = new Semaphore(1)

  def receive = {
    // Someone sent us a vector update
    case m: InternalMessages.CurrentVector => {
      val scaledDelta =
      currentAvg = currentAvg + m.v * (1.0/(others.size+1))
      newDataSemaphore.release()
    }

    case ppm: InternalMessages.PingPong => {
      println("new message from "+sender)
      sender ! "gotit!"
    }
    case m: InternalMessages.WakeupMsg => { println("activated local!"); sender ! "yo" }
    case s: String => println(s)
    case _ => println("hello, world!")
  }

  def shuttingDown: Receive = {
    case _ => println("GOT SHUTDOWN!")
  }

  def connectToOthers(allHosts: Array[String]) {
    var i = 0
    for(host <- allHosts) {
      if(!host.equals(address)) {
        others.put(i, context.actorSelection(allHosts(i)))
      }
      i += 1
    }
  }

  def sendPingPongs() {
    for(other <- others.values) {
      other ! new InternalMessages.PingPong
    }
  }

  def broadcastWeightVector(v: BV[Double]) {
    for(other <- others.values) {
      other ! new InternalMessages.CurrentVector(v)
    }
  }
}

// Set up per-partition communication network between workers
object CommSetup {
  def setup(input: RDD[(Double, Vector)]): RDD[WorkerCommunication] = {
    val workers: RDD[(WorkerCommunication)] = input.mapPartitions {
      iter =>
        val workerName = UUID.randomUUID().toString
        val address = Worker.HACKakkaHost+workerName
        val hack = new WorkerCommunicationHack()
        println(address)
        val aref= Worker.HACKworkerActorSystem.actorOf(Props(new WorkerCommunication(address, hack)), workerName)
        implicit val timeout = Timeout(15 seconds)

        val f = aref ? new InternalMessages.WakeupMsg
        Await.result(f, timeout.duration).asInstanceOf[String]

        Iterator(hack.ref)
    }

    val addresses = workers.map { w => w.address }.collect()

    workers.foreach {
      w => w.connectToOthers(addresses)
    }

    workers.foreach {
      w => w.sendPingPongs()
    }

    workers
  }
}

@DeveloperApi
class AsyncSGDLocalOptimizer(val gradient: Gradient,
                             val eta_0: Double = 1.0,
                             val maxLocalSolves: Int = 10,
                             val maxIterations: Int = Integer.MAX_VALUE,
                             val epsilon: Double = 0.001) {
  def apply(data: Array[(Double, Vector)], w0: BDV[Double], w_avg_input: BDV[Double], lambda: BDV[Double],
            rho: Double, comm: WorkerCommunication): BDV[Double] = {

    val w: BDV[Double] = w0.copy
    val nExamples = data.length

    comm.currentAvg = w_avg_input

    var solveNo = 0
    while(solveNo < maxLocalSolves) {
      comm.newDataSemaphore.acquireUninterruptibly()
      comm.newDataSemaphore.drainPermits()

      val w_avg = comm.currentAvg

      var t = 0
      var residual = Double.MaxValue

      val nExamples = data.length
      while (t < maxIterations && residual > epsilon) {
        val (label, features) = data(Random.nextInt(nExamples))
        val (gradLoss, loss) = gradient.compute(features, label, Vectors.fromBreeze(w))
        // compute the gradient of the full lagrangian
        val gradL = gradLoss.toBreeze.asInstanceOf[BDV[Double]] + lambda + (w - w_avg) * rho
        // Set the learning rate
        val eta_t = eta_0 / (t + 1)
        // w = w + eta_t * point_gradient
        axpy(-eta_t, gradL, w)
        // Compute residual
        residual = eta_t * norm(gradL, 2.0)
        t += 1
      }

      comm.broadcastWeightVector(w-w_avg)

      // TODO: recalculate lambda and rho here


      // Check the local prediction error:
      val propCorrect =
        data.map {
          case (y, x) => if (x.toBreeze.dot(w) * (y * 2.0 - 1.0) > 0.0) 1 else 0
        }
          .reduce(_ + _).toDouble / nExamples.toDouble
      println(s"Local prop correct: $propCorrect")
      println(s"Local iterations: ${t}")
    }

    // Return the final weight vector
    w
  }
}


class AsyncADMM private[mllib] extends Optimizer with Logging {

  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var epsilon: Double = 0.0
  private var localOptimizer: AsyncSGDLocalOptimizer = null

  /**
   * Set the number of iterations for ADMM. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the local optimizer to use for subproblems.
   */
  def setLocalOptimizer(opt: AsyncSGDLocalOptimizer): this.type = {
    this.localOptimizer = opt
    this
  }

  /**
   * Set the local optimizer to use for subproblems.
   */
  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }


  /**
   * Solve the provided convex optimization problem.
   */
  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    val blockData: RDD[Array[(Double, Vector)]] = data.mapPartitions(iter => Iterator(iter.toArray)).cache()
    val dim = blockData.map(block => block(0)._2.size).first()
    val nExamples = blockData.map(block => block.length).reduce(_+_)
    val numPartitions = blockData.partitions.length
    println(s"nExamples: $nExamples")
    println(s"dim: $dim")
    println(s"number of solver ${numPartitions}")

    var primalResidual = Double.MaxValue
    var dualResidual = Double.MaxValue
    var iter = 0
    var rho  = 0.0

    val commSystems = CommSetup.setup(data)

    // Make a zero vector
    var wAndLambda = blockData.map{ block =>
      val dim = block(0)._2.size
      (BDV.zeros[Double](dim), BDV.zeros[Double](dim))
    }
    var w_avg: BDV[Double] = BDV.zeros[Double](dim)

    val optimizer = localOptimizer

    // Compute w and new lambda
    wAndLambda = blockData.zipPartitions(wAndLambda, commSystems) {
      (dataIter, modelIter, commSystemIter) =>
      dataIter.zip(modelIter).zip(commSystemIter).map { case ((data, (w_old, lambda_old)), commSystem) =>
        // Update the lagrangian Multiplier by taking a gradient step
        val lambda: BDV[Double] = lambda_old + (w_old - w_avg) * rho
        val w = optimizer(data, w_old, w_avg, lambda, rho, commSystem)
        (w, lambda)
      }
    }.cache()
    // Compute new w_avg
    val new_w_avg = blockData.zipPartitions(wAndLambda) { (dataIter, modelIter) =>
      dataIter.zip(modelIter).map { case (data, (w, _)) => w * data.length.toDouble }
    }.reduce(_ + _) / nExamples.toDouble


    w_avg = new_w_avg

    println(s"Iteration: ${iter}")
    println(s"(Primal Resid, Dual Resid, Rho): ${primalResidual}, \t ${dualResidual}, \t ${rho}")

    Vectors.fromBreeze(w_avg)
  }
}