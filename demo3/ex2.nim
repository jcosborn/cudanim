import timing, cpugpuarray, qexLite/metaUtils, math

proc test(vecLen, memLen: static[int]; N: int) =
  var
    x = newColorMatrixArray(vecLen,memLen,N) # array of N 3x3 single prec complex matrices
    y = newColorMatrixArray(vecLen,memLen,N)
    z = newColorMatrixArray(vecLen,memLen,N)
    rep = 0                     # accumulates the number of runs

  let
    mr = float(3 * 8 * x.T.N * x.T.N * N) / float(1024 * 1024 * 1024) # Resident memory in 2^30 bytes
    mt = 4 * mr / 3           # Memory transaction
    fp = float(8 * x.T.N * x.T.N * x.T.N * N) * 1e-9 # Floating point op / 10^9
  template timeit(label:string, s:untyped) =
    var
      R {.global.} = 64         # Base repeat
      T {.global.} = 1.0        # Time limit
      t:float
    while true:
      t = timex(rep, R, s)
      threadSingle:
        R *= int(1+0.5/t)       # set up to run at least 0.5 sec
        T -= t
      if T < 0: break           # Repeat until time is up
    threadSingle:               # Use the last R & t for performance measure
      printf("%8d %2d %d %-8s rep: %7d KB: %6.0f ms: %6.3f GF/s: %6.2f GB/s: %6.2f\n",
             N, vecLen, memLen, label, R, 1024*1024*mr, 1e3*t/R.float, fp*R.float/t, mt*R.float/t)

  threads:                 # CPU threads
    x := 0                 # set them to diagonal matrices on CPU
    y := 1
    z := 2
    timeit "CPU": x += y * z

  timeit "GPU5": # includes kernel launching and synchronization
    onGpu(N, 32):       # Number of threads, threads per block
      x += y * z
  timeit "GPU6": onGpu(N, 64): x += y * z
  timeit "GPU7": onGpu(N, 128): x += y * z

  threads: timeit "CPU": x += y * z # back to CPU threads again

  if classify(x[100][1,1].re) == fcNan or abs(2-x[100][1,1].re/rep.float) > 1e-4:
    echo "ERROR"
    printf("# repeated: %7d  x[0][1,1]: %e\n", rep, x[100][1,1].re)
    quit 1
  x.free
  y.free
  z.free

for n in 8..26:
  staticFor v, 2, 7:
    when (1 shl v) >= (structsize(vectorizedElementType(float32)) div sizeof(float32)):
      staticFor ml, 1, 2:
        test(1 shl v, ml, 1 shl n)
