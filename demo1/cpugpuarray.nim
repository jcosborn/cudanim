import gpuarray
export gpuarray
import macros
import ../cuda
export cuda
import ../expr
import linalg
export linalg
include system/ansi_c

#template onGpu*(x: untyped): untyped = x
#template onGpu*(a,b,x: untyped): untyped = x

type
  ArrayObj*[T] = object
    p*: ptr array[0,T]
    n*: int
    g*: GpuArrayObj[T]
    lastOnGpu*: bool
    unifiedMem*: bool
  ArrayRef*[T] = ref ArrayObj[T]
  Array*[T] = ArrayRef[T]
  Arrays* = ArrayObj | ArrayRef
  Arrays2* = ArrayObj | ArrayRef
  Arrays3* = ArrayObj | ArrayRef

proc init[T](r: var ArrayObj[T], n: int) =
  var p: ptr T
  r.unifiedMem = true
  if r.unifiedMem:
    let err = cast[cint](cudaMallocManaged(cast[ptr pointer](addr p), n*sizeof(T)))
    if err != 0:
      echo "ERROR( ",err," ): cudaMallocManaged can't allocate ", n*sizeof(T)
      quit err
  else:
    p = createSharedU(T, n)
  r.n = n
  r.p = cast[type(r.p)](p)
proc init[T](r: var ArrayRef[T], n: int) =
  r.new
  r[].init(n)

proc free*[T](r: var ArrayObj[T]) =
  if r.unifiedMem: discard r.p.cudaFree
  else: discard r.g.p.cudaFree
proc free*[T](r: ArrayRef[T]) =
  if r.unifiedMem: discard r.p.cudaFree
  else: discard r.g.p.cudaFree

proc newArrayObj*[T](r: var ArrayObj[T], n: int) =
  r.init(n)
proc newArrayObj*[T](n: int): ArrayObj[T] =
  result.init(n)

proc newArrayRef*[T](r: var ArrayRef[T], n: int) =
  r.init(n)
proc newArrayRef*[T](n: int): ArrayRef[T] =
  result.init(n)

proc toGpu*(x: var Arrays) =
  if x.unifiedMem:
    if x.g.n==0:
      x.g.n = x.n
      x.g.p = cast[type(x.g.p)](x.p)
  else:
    if not x.lastOnGpu:
      x.lastOnGpu = true
      if x.g.n==0: x.g.init(x.n)
      let err = cudaMemcpy(x.g.p, x.p, x.n*sizeof(x.T), cudaMemcpyHostToDevice)
      if err: echo err

proc toCpu*(x: var Arrays) =
  if not x.unifiedMem:
    if x.lastOnGpu:
      x.lastOnGpu = false
      let err = cudaMemcpy(x.p, x.g.p, x.n*sizeof(x.T), cudaMemcpyDeviceToHost)
      if err: echo err

template getGpuPtr*(x: var Arrays): untyped =
  toGpu(x)
  x.g

template indexArray*(x: Arrays, i: SomeInteger): untyped =
  x.p[][i]
#template `[]=`(x: ArrayObj, i: SomeInteger, y: untyped): untyped =
#  x.p[][i] = y

macro indexArray*(x: Arrays{call}, y: SomeInteger): untyped =
  #echo "call[", y.repr, "]"
  #echo x.treerepr
  #if siteLocalsField.contains($x[0]):
  result = newCall(ident($x[0]))
  for i in 1..<x.len:
    let xi = x[i]
    result.add( quote do:
      indexArray(`xi`,`y`) )
  #else:
  #  result = quote do:
  #    let tt = `x`
  #    tt.d[`y`]
  #echo result.treerepr
  #echo result.repr

template `[]`*(x: ArrayObj, i: SomeInteger): untyped = indexArray(x, i)
#template `[]=`(x: ArrayObj, i: SomeInteger, y: untyped): untyped =
#  x.p[][i] = y

template `[]`*(x: ArrayRef, i: SomeInteger): untyped = indexArray(x, i)
#template `[]=`(x: ArrayRef, i: SomeInteger, y: untyped): untyped =
#  x.p[][i] = y

var threadNum* = 0
var numThreads* = 1
template getThreadNum*: untyped = threadNum
template getNumThreads*: untyped = numThreads
template `:=`*(x: Arrays, y: Arrays2) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  packVarsStmt((x,y), toCpu)
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] := y[i]
    i += nid

template `:=`*(x: Arrays, y: SomeNumber) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  packVarsStmt(x, toCpu)
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] := y
    i += nid

template `+=`*(x: Arrays, y: SomeNumber) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  packVarsStmt((x,y), toCpu)
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] += y
    i += nid

template `+=`*(x: Arrays, y: Arrays2) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  packVarsStmt((x,y), toCpu)
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] += y[i]
    i += nid

proc `+`*(x: Arrays, y: Arrays2): auto =
  when x is ArrayObj:
    var r: ArrayObj[type(x[0]+y[0])]
  else:
    var r: ArrayRef[type(x[0]+y[0])]
  echo "+\n"
  r
proc `*`*(x: Arrays, y: Arrays2): auto =
  when x is ArrayObj:
    var r: ArrayObj[type(x[0]*y[0])]
  else:
    var r: ArrayRef[type(x[0]*y[0])]
  echo "*\n"
  r

template newColorMatrixArray*(n: int): untyped =
  newArrayRef[Colmat[3,float32]](n)
template newComplexArray*(n: int): untyped =
  newArrayRef[Complex[float32]](n)
template newFloatArray*(n: int): untyped =
  newArrayRef[float32](n)

proc printf*(frmt: cstring): cint {.
  importc: "printf", header: "<stdio.h>", varargs, discardable.}

when isMainModule:
  var N = 100

  proc testfloat =
    var x = newArrayRef[float32](N)
    var y = newArrayRef[float32](N)
    var z = newArrayRef[float32](N)
    x := 1
    y := 2
    z := 3
    x += y * z
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i]: %g\n", x.n-1, x[x.n-1])
    onGpu(1,32):
      x += y * z
      if (x.n-1) mod getNumThreads() == getThreadNum():
        cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
        cprintf("x[%i]: %g\n", x.n-1, x[x.n-1])
    x.toCpu
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i]: %g\n", x.n-1, x[x.n-1])
  testfloat()

  proc testcomplex =
    var x = newArrayRef[Complex[float32]](N)
    var y = newArrayRef[Complex[float32]](N)
    var z = newArrayRef[Complex[float32]](N)
    x := 1
    y := 2
    z := 3
    x += y * z
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i]: %g\n", x.n-1, x[x.n-1].re)

    onGpu:
      x += y * z
      x += 1

    x += y * z
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i]: %g\n", x.n-1, x[x.n-1].re)
  testcomplex()

  proc testcolmat =
    var x = newArrayRef[Colmat[3,float32]](N)
    var y = newArrayRef[Colmat[3,float32]](N)
    var z = newArrayRef[Colmat[3,float32]](N)
    x := 1
    y := 2
    z := 3
    x += y * z
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i][0,0]: %g\n", x.n-1, x[x.n-1].d[0][0].re)

    onGpu(N):
      x += y * z

    x += y * z
    if (x.n-1) mod getNumThreads() == getThreadNum():
      cprintf("thread %i/%i\n", getThreadNum(), getNumThreads())
      cprintf("x[%i][0,0]: %g\n", x.n-1, x[x.n-1].d[0][0].re)
    x.free
    y.free
    z.free
  testcolmat()
