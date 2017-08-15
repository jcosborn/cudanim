import cpugpuarray
import qexLite/metaUtils
include system/timers

proc test(vecLen, memLen: static[int]; N: int) =
  var x = newColorMatrixArray(vecLen,memLen,N)
  var y = newColorMatrixArray(vecLen,memLen,N)
  var z = newColorMatrixArray(vecLen,memLen,N)

  template timeit(s:string, b:untyped) =
    const R = 128
    b # for warming up
    let t0 = getTicks()
    for i in 0..<R: b
    let t = float(getTicks()-t0)
    if getThreadNum()==0:
      let n = x.T.N
      let v = n*n*N*R
      printf("%8lld\t%3d %d\t%-7s\tmsec: %6.3f\tGF/s: %6.2f\tGB/s: %6.2f\n",
             N, vecLen, memLen, s,
             t*1e-6/R.float, float(8*n*v)/t, float(4*8*v)/t)

  threads:                 # CPU threads
    # set them to diagonal matrices on CPU
    x := 1
    y := 2
    z := 3

    # do something on CPU
    timeit "CPU":
      x += y * z

  timeit "GPU5":
    onGpu(N, 32):       # Number of threads, threads per block
      x += y * z
  timeit "GPU6":
    onGpu(N, 64):
      x += y * z
  timeit "GPU7":
    onGpu(N, 128):
      x += y * z

  threads:                 # CPU threads
    # do something on CPU again
    timeit "CPU":
      x += y * z

  x.free
  y.free
  z.free

for n in 10..26:
  staticFor v, 2, 7:
    when (1 shl v) >= (structsize(vectorizedElementType(float32)) div sizeof(float32)):
      staticFor ml, 1, 2:
        test(1 shl v, ml, 1 shl n)
