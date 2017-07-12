import cpugpuarray
include system/timers
include system/ansi_c
import strUtils

proc test =
  let N = 100000
  #var x = newFloatArray(N)
  #var y = newFloatArray(N)
  #var z = newFloatArray(N)
  #var x = newComplexArray(N)
  #var y = newComplexArray(N)
  #var z = newComplexArray(N)
  var x = newColorMatrixArray(N)
  var y = newColorMatrixArray(N)
  var z = newColorMatrixArray(N)

  var t0,t1: Ticks
  template tic =
    t0 = getTicks()
  template toc =
    t1 = getTicks()
    #echo "nanos: ", formatFloat((t1-t0).float, precision=0)
    cprintf("nanos:   %9i\n", t1-t0)
    #cprintf("GF/s: %9.3f\n", (2*N).float/(t1-t0).float)
    #cprintf("GF/s: %9.3f\n", (8*N).float/(t1-t0).float)
    #cprintf("GF/s: %9.3f\n", (3*72*N).float/(t1-t0).float)
    let n = x.T.N
    cprintf("GF/s: %9.3f\n", (8*n*n*n*N).float/(t1-t0).float)

  # set them to diagonal matrices on CPU
  x := 1
  y := 2
  z := 3

  # do something on CPU
  tic()
  x += y * z
  toc()
  tic()
  x += y * z
  toc()
  #for i in 1..10000:
  #  tic()
  #  x += y * z
  #  toc()

  var s = 1.0'f32
  template getGpuPtr(x: float): float = x
  # do something on GPU
  echo "GPU1"
  tic()
  onGpu:
    #var t = s
    x += y * z
    #if ff(): discard
      #z := 4
  toc()
  echo "GPU2"
  tic()
  onGpu(2*768,64):
    x += y * z
  #  #z := 4
  toc()

  # do something on CPU again
  tic()
  x += y * z
  toc()
  tic()
  x += y * z
  toc()

  #if x[0][0,0].re == 21.0:
  #  echo "yay, it worked!"
  #  echo "do you agree, GPU?"

  #onGpu:
  #  if getThreadNum()==0:
  #    if x[0][0,0].re == 21.0:
  #      printf("yes, I agree!\n")

  # outputs:
  #   yay, it worked!
  #   do you agree, GPU?
  #   yes, I agree!

test()
