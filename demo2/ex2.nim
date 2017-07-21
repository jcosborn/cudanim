import cpugpuarray
include system/timers
include system/ansi_c
import strUtils

proc test(N: int) =
  var x = newColorMatrixArray(N)
  var y = newColorMatrixArray(N)
  var z = newColorMatrixArray(N)

  template timeit(s:string, b:untyped) =
    let R = 64
    let t0 = getTicks()
    for i in 0..<R: b
    let t1 = getTicks()
    let n = x.T.N
    cprintf("%8lld\t%-7s\tmsec: %7.3f\tGF/s: %6.3f\n", N, s, (t1-t0).float*1e-6/R.float, (8*n*n*n*N*R).float/(t1-t0).float)

  # set them to diagonal matrices on CPU
  x := 1
  y := 2
  z := 3

  # do something on CPU
  timeit "CPU":
    x += y * z

  var s = 1.0'f32
  template getGpuPtr(x: float): float = x
  # do something on GPU
  timeit "GPU1":
    onGpu(1 shl 10,1 shl 4):
      #var t = s
      x += y * z
      #if ff(): discard
        #z := 4
  timeit "GPU2":
    onGpu(1 shl 10,1 shl 5):
      x += y * z
  timeit "GPU3":
    onGpu(1 shl 10,1 shl 6):
      x += y * z
  timeit "GPU4":
    onGpu(1 shl 10,1 shl 7):
      x += y * z

  # do something on CPU again
  timeit "CPU":
    x += y * z

  x.free
  y.free
  z.free

#for n in 10..25: test(1 shl n)    # 7 GB ~ float su3 × 3 × 2^25
for n in 10..24: test(1 shl n)
