package edu.berkeley.emerson

import edu.berkeley.emerson.Emerson.Params

import breeze.linalg.{max, norm, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.util.Random
import scala.util.hashing.MurmurHash3




object DataLoaders {
  def loadBismark(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    val data = sc.textFile(filename, params.numPartitions)
      .filter(s => !s.isEmpty && s(0) == '{')
      .map(s => s.split('\t'))
      .map {
      case Array(x, y) =>
        val features = x.stripPrefix("{").stripSuffix("}").split(',').map(xi => xi.toDouble)
        val label = if (y.toDouble > 0) 1 else 0
        LabeledPoint(label, new DenseVector(features))
    }.cache()

    data
  }

  def makeDictionary(colId: Int, tbl: RDD[Array[String]]): Map[String, Int] = {
    tbl.map(row => row(colId)).distinct.collect.zipWithIndex.toMap
  }

  def makeBinary(value: String, dict: Map[String, Int]): Array[Double] = {
    val array = new Array[Double](dict.size)
    array(dict(value)) = 1.0
    array
  }

  def loadDBLP(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    println("loading data!")
    val rawData = sc.textFile(filename, params.numPartitions)

    var maxFeatureID = params.inputTokenHashKernelDimension
    if (params.inputTokenHashKernelDimension < 0) {
      maxFeatureID = rawData.map(line => max(line.split(' ').tail.map(s => s.toInt))).max()
    }

    rawData.map(line => {
      val splits = line.split(' ')
      val year = splits(0).toInt
      val label = if (year < params.dblpSplitYear) 0.0 else 1.0
      val features: Array[Double] = Array.fill[Double](maxFeatureID)(0.0)
      var i = 1
      while (i < splits.length) {
        val hc: Int = MurmurHash3.stringHash(splits(i))
        features(Math.abs(hc) % features.length) += (if (hc > 0) 1.0 else -1.0)
        i += 1
      }
      LabeledPoint(label, new DenseVector(features))
    }).repartition(params.numPartitions)
  }

  def loadWikipedia(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    println("loading data!")
    val rawData = sc.textFile(filename, params.numPartitions)

    var maxFeatureID = params.inputTokenHashKernelDimension
    if (params.inputTokenHashKernelDimension < 0) {
      maxFeatureID = rawData.map(line => max(line.split(' ').tail.map(s => s.toInt))).max()
    }

    val labelStr = params.wikipediaTargetWordToken.toString

    rawData.map(line => {
      val splits = line.split(' ')
      val features: Array[Double] = Array.fill[Double](maxFeatureID)(0.0)
      var i = 1
      var labelFound = false
      while (i < splits.length) {
        if(splits(i).equals(labelStr)) {
          labelFound = true
        } else {
          val hc: Int = MurmurHash3.stringHash(splits(i))
          features(Math.abs(hc) % features.length) += (if (hc > 0) 1.0 else -1.0)
        }
        i += 1
      }
      LabeledPoint(if(labelFound) 1.0 else 0.0, new DenseVector(features))
    }).repartition(params.numPartitions)
  }

  private def hrMinToScaledHr(s: String): Double = {
    if(s.length == 4) {
      return s.take(2).toDouble/24.
    } else {
      assert(s.length == 3)
      return s.take(1).toDouble/24.
    }
  }

  def loadFlights(sc: SparkContext, filename: String, params: Params): RDD[LabeledPoint] = {
    val labels = Array("Year", "Month", "DayOfMonth", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime",
      "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime",
      "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiIn", "TaxiOut",
      "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay",
      "NASDelay", "SecurityDelay", "LateAircraftDelay").zipWithIndex.toMap
    println("Loading data")
    val rawData = sc.textFile(filename, params.numPartitions).
      filter(s => !s.contains("Year")).
      map(s => s.split(",")).cache()

    val carrierDict = makeDictionary(labels("UniqueCarrier"), rawData)
    val flightNumDict = makeDictionary(labels("FlightNum"), rawData)
    val tailNumDict = makeDictionary(labels("TailNum"), rawData)
    val originDict = makeDictionary(labels("Origin"), rawData)
    val destDict = makeDictionary(labels("Dest"), rawData)

    val data = rawData.map {
      row =>
        val value_arr = Array.fill(12)(1.0)
        val idx_arr = new Array[Int](12)

        var idx_offset = 0
        for(i <- 1 until 5) {
          var v: Double = 0

          if (row(i) != "N/A") {
            v = row(i).toDouble

            // month
            if (i == 1) {
              v /= 12
            }

            // day of month
            if (i == 2) {
              v /= 31
            }

            // day of week
            if (i == 3) {
              v /= 7
            }

            // deptime
            if (i == 4) {
              v = hrMinToScaledHr(row(i))
            }
          }

          value_arr(idx_offset) = v
          idx_arr(idx_offset) = idx_offset
          idx_offset += 1
        }

        var bitvector_offset = idx_offset

        idx_arr(idx_offset) = bitvector_offset + carrierDict(row(labels("UniqueCarrier")))
        idx_offset += 1
        bitvector_offset += carrierDict.size

        idx_arr(idx_offset) = bitvector_offset + flightNumDict(row(labels("FlightNum")))
        idx_offset += 1
        bitvector_offset += flightNumDict.size

      /*
        idx_arr(idx_offset) = bitvector_offset + tailNumDict(row(labels("TailNum")))
        idx_offset += 1
        bitvector_offset += tailNumDict.size
        */

        idx_arr(idx_offset) = bitvector_offset + originDict(row(labels("Origin")))
        idx_offset += 1
        bitvector_offset += originDict.size

        idx_arr(idx_offset) = bitvector_offset + destDict(row(labels("Dest")))
        idx_offset += 1
        bitvector_offset += destDict.size

        // add one for bias term
        bitvector_offset += 1

        val delay = row(labels("ArrDelay"))
        val label = if (delay != "NA" && delay.toDouble > 0) 1.0 else 0.0

        assert(idx_offset == 11)

        LabeledPoint(label, new SparseVector(bitvector_offset, idx_arr, value_arr))
    }

    val ret = data.repartition(params.numPartitions)
    ret.cache().count()
    rawData.unpersist(true)

    ret
  }

  /*
    Build a dataset of points drawn from one of two 'point clouds' (one at [5,...] and one at [10, ...])
    with either all positive or all negative labels.

    labelNoise controls how many points within each cloud are mislabeled
    cloudSize controls the radius of each cloud
    partitionSkew controls how much of each cloud is visible to each partition
      partitionSkew = 0 means each partition only sees one cloud
      partitionSkew = .5 means each partition sees half of each cloud
   */
  def generatePairCloud(sc: SparkContext,
                        dim: Int,
                        labelNoise: Double,
                        cloudSize: Double,
                        partitionSkew: Double,
                        numPartitions: Int,
                        pointsPerPartition: Int): RDD[LabeledPoint] = {
    sc.parallelize(1 to numPartitions, numPartitions).flatMap { idx =>
      val plusCloud = new DenseVector(Array.fill[Double](dim)(10.0))
      plusCloud.values(dim - 1) = 1
      val negCloud = new DenseVector(Array.fill[Double](dim)(5.0))
      negCloud.values(dim - 1) = 1

      // Seed the generator with the partition index
      val random = new Random(idx)
      val isPartitionPlus = idx % 2 == 1

      (0 until pointsPerPartition).iterator.map { pt =>
        val isPointPlus = if (random.nextDouble() < partitionSkew) isPartitionPlus else !isPartitionPlus
        val trueLabel: Double = if (isPointPlus) 1.0 else 0.0

        val pointCenter = if (isPointPlus) plusCloud else negCloud

        // calculate the actual point in the cloud
        val chosenPoint = new DenseVector(new Array[Double](dim))
        for (d <- 0 until dim - 1) {
          chosenPoint.values(d) = pointCenter.values(d) + random.nextGaussian() * cloudSize
        }
        chosenPoint.values(dim - 1) = 1.0

        val chosenLabel = if (random.nextDouble() < labelNoise) (trueLabel+1) % 2 else trueLabel

        new LabeledPoint(chosenLabel, chosenPoint)
      }
    }
  }
}
