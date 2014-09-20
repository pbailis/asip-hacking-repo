package edu.berkeley.emerson

import breeze.linalg.{max, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import edu.berkeley.emerson.Emerson.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD

import scala.util.Random
import scala.util.hashing.MurmurHash3



object DataLoaders {
  def loadBismark(sc: SparkContext, filename: String, params: Params): RDD[Array[(Double, BV[Double])]] = {
    val data = sc.textFile(filename, params.numPartitions)
      .filter(s => !s.isEmpty && s(0) == '{')
      .map(s => s.split('\t'))
      .map {
      case Array(x, y) =>
        val features = x.stripPrefix("{").stripSuffix("}").split(',').map(xi => xi.toDouble)
        val label = if (y.toDouble > 0) 1.0 else 0.0
        val xFeatures: BV[Double] = new BDV[Double](features)
        (label, xFeatures)
    }.repartition(params.numPartitions).mapPartitions(iter => Iterator(iter.toArray)).cache()
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

  def loadDBLP(sc: SparkContext, filename: String, params: Params): RDD[Array[(Double, BV[Double])]] = {
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
      val x: BV[Double] = new BDV[Double](features)
      (label, x)
    }).repartition(params.numPartitions).mapPartitions(iter => Iterator(iter.toArray)).cache()
  }

  def loadWikipedia(sc: SparkContext, filename: String, params: Params):
      RDD[Array[(Double, BV[Double])]] = {
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
      val label = if(labelFound) 1.0 else 0.0
      val x: BV[Double] = new BDV[Double](features)
      (label, x)
    }).repartition(params.numPartitions).mapPartitions(iter => Iterator(iter.toArray)).cache()
  }

  private def hrMinToScaledHr(s: String): Double = {
    var r = s.toDouble
    val mins = r % 60
    (((r-mins)/100*60)+mins)/60/24
  }

  def loadFlights(sc: SparkContext, filename: String, params: Params):
      RDD[Array[(Double, BV[Double])]] = {
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
        val value_arr = Array.fill(10)(1.0)
        val idx_arr = new Array[Int](10)

        var idx_offset = 0
        for(i <- 1 until 5) {
          var v: Double = 0

          if (row(i) != "NA") {
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

        val x: BV[Double] = new BSV[Double](idx_arr, value_arr, bitvector_offset)
        (label, x)
    }.repartition(params.numPartitions).mapPartitions(iter => Iterator(iter.toArray)).cache()

    data.count()
    rawData.unpersist(true)
    data
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
                        pointsPerPartition: Int): RDD[Array[(Double, BV[Double])]] = {
    sc.parallelize(1 to numPartitions, numPartitions).map { idx =>
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
        val chosenPoint: BV[Double] = BDV.zeros[Double](dim)
        for (d <- 0 until dim - 1) {
          chosenPoint(d) = pointCenter.values(d) + random.nextGaussian() * cloudSize
        }
        chosenPoint(dim - 1) = 1.0

        val chosenLabel: Double = if (random.nextDouble() < labelNoise) (trueLabel+1) % 2 else trueLabel

        (chosenLabel, chosenPoint)
      }.toArray
    }
  }


  def elementMax(a: BV[Double], b: BV[Double]): BV[Double] = {
    val res = a.toDenseVector
    var i = 0
    val n = a.length
    while (i < n) {
      res(i) = math.max(a(i), b(i))
      i += 1
    }
    res
  }

  def normalizeData(data: RDD[Array[(Double, BV[Double])]]): RDD[Array[(Double, BV[Double])]] = {
    val nExamples = data.map { data => data.length }.reduce( _ + _ )

    val xmax: BV[Double] = data.map {
      data => data.view.map { case (y, x) => x }.reduce((a,b) => elementMax(a,b))
    }.reduce((a,b) => elementMax(a,b))
//
//    val xbar: BV[Double] = data.map {
//      data => data.view.map { case (y, x) => x }.reduce(_ + _)
//    }.reduce(_ + _) / nExamples.toDouble
//
//    val xxbar: BV[Double] = data.map {
//      data => data.view.map { case (y, x) => x :* x }.reduce(_ + _)
//    }.reduce(_ + _) / nExamples.toDouble
//
//    val variance: BDV[Double] = (xxbar - (xbar :* xbar)).toDenseVector
//
//    val stdev: BDV[Double] = breeze.numerics.sqrt(variance)
//
//    // Just in case there are constant columns set the standard deviation to 1.0
//    for(i <- 0 until stdev.size) {
//      if (stdev(i) == 0.0) { stdev(i) = 1.0 }
//    }

//    assert(xbar.size == stdev.size)

    // Just in case there are constant columns set the standard deviation to 1.0
    for(i <- 0 until xmax.size) {
      if (xmax(i) == 0.0) { xmax(i) = 1.0 }
    }

    val data2 = data.map { data =>
      data.map { case (y, x) =>
        assert(x.size == xmax.size)
        // ugly hack where I reuse the vector
//        x -= xbar
//        x /= stdev
        x /= xmax
        (y, x)
      }
    }.cache()
    data2.foreach( x => () )
    data.unpersist()
    data2
  }

}
