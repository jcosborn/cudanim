import cpugpuarray
include system/timers
include system/ansi_c
import strUtils

proc test(N: int) =
  #var x = newFloatArray(N)
  #var y = newFloatArray(N)
  #var z = newFloatArray(N)
  #var x = newComplexArray(N)
  #var y = newComplexArray(N)
  #var z = newComplexArray(N)
  var x = newColorMatrixArray(N)
  var y = newColorMatrixArray(N)
  var z = newColorMatrixArray(N)

  template timeit(s:string, b:untyped) =
    var t0 = getTicks()
    b
    var t1 = getTicks()
    let t = if s.len == 0: $N&"\t" else: $N&"\t"&s&"\t"
    #echo "nanos: ", formatFloat((t1-t0).float, precision=0)
    cprintf(t&"nanos:   %9i\n", t1-t0)
    #cprintf("GF/s: %9.3f\n", (2*N).float/(t1-t0).float)
    #cprintf("GF/s: %9.3f\n", (8*N).float/(t1-t0).float)
    #cprintf("GF/s: %9.3f\n", (3*72*N).float/(t1-t0).float)
    let n = x.T.N
    cprintf(t&"GF/s: %9.3f\n", (8*n*n*n*N).float/(t1-t0).float)

  # set them to diagonal matrices on CPU
  x := 1
  y := 2
  z := 3

  # do something on CPU
  timeit "CPU":
    x += y * z
  timeit "CPU":
    x += y * z
  #for i in 1..10000:
  #  tic()
  #  x += y * z
  #  toc()

  var s = 1.0'f32
  template getGpuPtr(x: float): float = x
  # do something on GPU
  timeit "GPU1":
    onGpu(2*768,64):
      #var t = s
      x += y * z
      #if ff(): discard
        #z := 4
  timeit "GPU2":
    onGpu(2*768,64):
      x += y * z
  timeit "GPU3":
    onGpu(1 shl 20,1 shl 10):
      x += y * z

  # do something on CPU again
  timeit "CPU":
    x += y * z
  timeit "CPU":
    x += y * z

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

for n in 10..24: test(1 shl n)
