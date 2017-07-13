when not declared(haveCuda):
  const haveCuda = true

when haveCuda:
  import ../cuda

import macros
include system/ansi_c
import linalg

type
  GpuArrayObj*[T] = object
    p*: ptr array[0,T]
    n*: int
  GpuArrayRef*[T] = ref GpuArrayObj[T]
  GpuArray*[T] = GpuArrayRef[T]
  GpuArrays* = GpuArrayObj | GpuArrayRef
  GpuArrays2* = GpuArrayObj | GpuArrayRef
  GpuArrays3* = GpuArrayObj | GpuArrayRef

proc init*[T](r: var GpuArrayObj[T], n: int) =
  var p: pointer
  when haveCuda:
    let err = cudaMalloc(p.addr, n*sizeof(T))
    if err:
      echo "alloc err: ", err
      quit(-1)
  else:
    p = createSharedU(T, n)
  r.n = n
  r.p = cast[type(r.p)](p)
proc init[T](r: var GpuArrayRef[T], n: int) =
  r.new
  r[].init(n)

proc newGpuArrayObj*[T](r: var GpuArrayObj[T], n: int) =
  r.init(n)
proc newGpuArrayObj*[T](n: int): GpuArrayObj[T] =
  result.init(n)

proc newGpuArrayRef*[T](r: var GpuArrayRef[T], n: int) =
  r.init(n)
proc newGpuArrayRef*[T](n: int): GpuArrayRef[T] =
  result.init(n)

template getGpuPtr*(x: SomeNumber): untyped = x
#template getGpuPtr(x: GpuArrayObj): untyped = x
template getGpuPtr*(x: GpuArrayRef): untyped = x[]
#template getGpuPtr(x: GpuArrayRef): untyped = x.p
#template getGpuPtr(x: GpuArrayRef): untyped = (p:x.p,n:x.n)

template indexGpuArray*(x: GpuArrays, i: SomeInteger): untyped =
  x.p[][i]

macro indexGpuArray*(x: GpuArrays{call}, y: SomeInteger): untyped =
  #echo "call[", y.repr, "]"
  #echo x.treerepr
  #if siteLocalsField.contains($x[0]):
  result = newCall(ident($x[0]))
  for i in 1..<x.len:
    let xi = x[i]
    result.add( quote do:
      indexGpuArray(`xi`,`y`) )
  #else:
  #  result = quote do:
  #    let tt = `x`
  #    tt.d[`y`]
  #echo result.treerepr
  #echo result.repr

template `[]`*(x: GpuArrayObj, i: SomeInteger): untyped = indexGpuArray(x, i)
template `[]=`*(x: GpuArrayObj, i: SomeInteger, y: untyped): untyped =
  x.p[][i] = y

template `[]`*(x: GpuArrayRef, i: SomeInteger): untyped =
  echo "GAR[]"
  x.p[][i]
template `[]=`*(x: GpuArrayRef, i: SomeInteger, y: untyped): untyped =
  x.p[][i] = y

var threadNum = 0
var numThreads = 1
template getThreadNum: untyped = threadNum
template getNumThreads: untyped = numThreads
template `:=`*(x: GpuArrays, y: GpuArrays2) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] := y[i]
    i += nid

template `:=`*(x: GpuArrays, y: SomeNumber) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  while i<x.n:
    x[i] := y
    #echo i, "/", x.n
    i += nid

template `+=`*(x: GpuArrays, y: SomeNumber) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  #cprintf("%i/%i\n", i, x.n)
  while i<x.n:
    x[i] += y
    #cprintf("%i/%i\n", i, x.n)
    i += nid

template `+=`*(x: GpuArrays, y: GpuArrays2) =
  #cprintf("t %i/%i  b %i/%i\n", getThreadIdx(), getThreadDim(), getBlockIdx(), getBlockDim())
  #let i = getBlockDim().x * getBlockIdx().x + getThreadIdx().x
  mixin getThreadNum, getNumThreads
  let tid = getThreadNum()
  let nid = getNumThreads()
  var i = tid
  #cprintf("%i/%i\n", i, x.n)
  while i<x.n:
    x[i] += y[i]
    #cprintf("%i/%i\n", i, x.n)
    i += nid

proc `+`*(x: GpuArrays, y: GpuArrays2): auto =
  when x is GpuArrayObj:
    var r: GpuArrayObj[type(x[0]+y[0])]
  else:
    var r: GpuArrayRef[type(x[0]+y[0])]
  cprintf("+\n")
  r
proc `*`*(x: GpuArrays, y: GpuArrays2): auto =
  when x is GpuArrayObj:
    var r: GpuArrayObj[type(x[0]*y[0])]
  else:
    var r: GpuArrayRef[type(x[0]*y[0])]
  cprintf("*\n")
  r

when isMainModule:
  var N = 1000

  proc testfloat =
    var x = newGpuArrayRef[float32](N)
    var y = newGpuArrayRef[float32](N)
    var z = newGpuArrayRef[float32](N)
    #cprintf("x.n: %i\n", x.n)
    onGpu(1,32):
      x += y * z
  testfloat()

  proc testcomplex =
    var x = newGpuArrayRef[Complex[float32]](N)
    var y = newGpuArrayRef[Complex[float32]](N)
    var z = newGpuArrayRef[Complex[float32]](N)
    onGpu(N):
      x += y * z
  testcomplex()

  proc testcolmat =
    var x = newGpuArrayRef[Colmat[3,float32]](N)
    var y = newGpuArrayRef[Colmat[3,float32]](N)
    var z = newGpuArrayRef[Colmat[3,float32]](N)
    #y := 1
    #z := 2
    onGpu(N):
      x += y * z
  testcolmat()
