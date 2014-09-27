package org.apache.spark.mllib.evaluation

object Straggler {
  var interval = 1000
  var napDuration = 2000
  var isStraggler = false
  @volatile var napTime = true
 
  class Trigger extends Thread {
    override def run() {
      Thread.sleep(interval)
      napTime = true
    }
  }

  def straggle() {
    if (isStraggler && napTime) {
      // Try to clear the flag
      var iSleep = false
      this.synchronized {
        if (napTime) {
          iSleep = true
          napTime = false
        }
      }
      if (iSleep) {
        println("Straggling")
        Thread.sleep(napDuration)
        new Trigger().start()
      }
      // println("Straggling")
      // Thread.sleep(napDuration)
      // // Launch the trigger thread again
      // this.synchronized {
      //   if (napTime) {
      //     napTime = false
      //     new Trigger().start()
      //   }
      // }
    }
  }


}
