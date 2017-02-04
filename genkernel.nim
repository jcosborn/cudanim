import cuda

type gpuArray[T] = distinct ptr array[0,T]
template `[]`(x: gpuArray, i: SomeInteger): untyped =
  (ptr array[0,x.T])(x)[][i]
template `[]=`(x: gpuArray, i: SomeInteger, y: untyped): untyped =
  (ptr array[0,x.T])(x)[][i] = y

proc alloc(a: var gpuArray, n: int) =
  let err = cudaMalloc(a, n*sizeof(a.T))
  echo "alloc err: ", err
proc newGpuArray[T](n: int): gpuArray[T] =
  var p: pointer
  let err = cudaMalloc(p.addr, n*sizeof(T))
  let pa = cast[ptr array[0,T]](p)
  result = (type(result))(pa)
  if err:
    echo "alloc err: ", err
    quit(-1)

proc timesTwo*(a: gpuArray; n: int32) {.cudaGlobal.} =
  var i = blockDim.x * blockIdx.x + threadIdx.x
  if i < n:
    #a[i] = a.T(2) * a[i]
    a[i] *= a.T(2)

var
  n = 10000.int32
  a = newGpuArray[float32](n)
  b = newGpuArray[float64](n)

var threadsPerBlock: int32 = 256
var blocksPerGrid: int32 = (n + threadsPerBlock - 1) div threadsPerBlock

timesTwo<<(blocksPerGrid,threadsPerBlock)>>(a,n)

timesTwo<<(blocksPerGrid,threadsPerBlock)>>(b,n)
