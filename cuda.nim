import macros
#import deviceProcGen
import inline
#macro dumpType(x:typed): auto =
#  result = newEmptyNode()
#  echo x.getType.treerepr
proc addChildrenFrom*(dst,src: NimNode): NimNode =
  for c in src: dst.add(c)
  result = dst
macro procInst*(p: typed): auto =
  #echo "begin procInst:"
  #echo p.treerepr
  result = p[0]
macro makeCall*(p: proc, x: tuple): NimNode =
  result = newCall(p).addChildrenFrom(x)

type
  CudaDim3* {.importc:"dim3",header:"cuda_runtime.h".} = object
    x*, y*, z*: cint
  cudaError_t* {.importc,header:"cuda_runtime.h".} = object
  cudaMemcpyKind* {.importc,header:"cuda_runtime.h".} = object
var
  cudaSuccess*{.importC,header:"cuda_runtime.h".}: cudaError_t
  cudaMemcpyHostToDevice*{.importC,header:"cuda_runtime.h".}: cudaMemcpyKind
  cudaMemcpyDeviceToHost*{.importC,header:"cuda_runtime.h".}: cudaMemcpyKind

#template toPointer*(x: pointer): pointer = x
#template toPointer*[T](x: ptr T): pointer = pointer(x)
#template toPointer*(x: seq): pointer = toPointer(x[0])
#template toPointer*(x: not (pointer|seq)): pointer = pointer(unsafeAddr(x))
template toPointer*(x: typed): pointer =
  #dumpType: x
  when x is pointer: x
  elif x is ptr: x
  elif x is seq: toPointer(x[0])
  else: pointer(unsafeAddr(x))
template dataAddr*(x: typed): pointer =
  #dumpType: x
  when x is seq: dataAddr(x[0])
  elif x is array: dataAddr(x[0])
  else: pointer(unsafeAddr(x))

proc cudaGetLastError*(): cudaError_t
  {.importC,header:"cuda_runtime.h".}
proc cudaGetErrorStringX*(error: cudaError_t): ptr char
  {.importC:"cudaGetErrorString",header:"cuda_runtime.h".}
proc cudaGetErrorString*(error: cudaError_t): cstring =
  var s {.codegendecl:"const $# $#".} = cudaGetErrorStringX(error)
  result = s
proc `$`*(error: cudaError_t): string =
  let s = cudaGetErrorString(error)
  result = $s
converter toBool*(e: cudaError_t): bool =
  e != cudaSuccess

proc cudaMalloc*(p:ptr pointer, size: csize): cudaError_t
  {.importC,header:"cuda_runtime.h".}
template cudaMalloc*(p:pointer, size: csize): cudaError_t =
  cudaMalloc((ptr pointer)(p.addr), size)
proc cudaFree*(p: pointer): cudaError_t
  {.importC,header:"cuda_runtime.h".}

proc cudaMemcpyX*(dst,src: pointer, count: csize, kind: cudaMemcpyKind):
  cudaError_t {.importC:"cudaMemcpy",header:"cuda_runtime.h".}
template cudaMemcpy*(dst,src: typed, count: csize,
                     kind: cudaMemcpyKind): cudaError_t =
  let pdst = toPointer(dst)
  let psrc = toPointer(src)
  cudaMemcpyX(pdst, psrc, count, kind)

proc cudaLaunchKernel(p:pointer, gd,bd: CudaDim3, args: ptr pointer)
  {.importC,header:"cuda_runtime.h".}

proc cudaDeviceReset*(): cudaError_t
  {.importC,header:"cuda_runtime.h".}

#proc printf*(fmt:cstring):cint {.importc,varargs,header:"<stdio.h>",discardable.}
#proc fprintf*(stream:ptr FILE,fmt:cstring):cint {.importc,varargs,header:"<stdio.h>".}
#proc malloc*(size: csize):pointer {.importc,header:"<stdlib.h>".}

template cudaDefs(body: untyped): untyped {.dirty.} =
  var blockDim{.global,importC,noDecl.}: CudaDim3
  var blockIdx{.global,importC,noDecl.}: CudaDim3
  var threadIdx{.global,importC,noDecl.}: CudaDim3
  template `[]`[T](x: ptr T, i: SomeInteger): untyped =
    cast[ptr array[0,T]](x)[][i]
  template `[]=`[T](x: ptr T, i: SomeInteger, y:untyped): untyped =
    cast[ptr array[0,T]](x)[][i] = y
  #bind deviceProcGen
  #deviceProcGen:
  bind inlineProcs
  inlineProcs:
    body

template cudaLaunch*(p: proc, nb,nt: SomeInteger,
                     arg: varargs[pointer,dataAddr]) =
  var pp: proc = p
  var gridDim, blockDim: CudaDim3
  gridDim.x = blocksPerGrid
  gridDim.y = 1
  gridDim.z = 1
  blockDim.x = threadsPerBlock
  blockDim.y = 1
  blockDim.z = 1
  var args: array[arg.len, pointer]
  for i in 0..<arg.len: args[i] = arg[i]
  cudaLaunchKernel(pp, gridDim, blockDim, addr args[0])

#macro `<<`*(x:varargs[untyped]): auto =
#  result = newEmptyNode()
#  echo x.treerepr
template `<<`*(p: proc, x: tuple): untyped = (p,x)
template getInst*(p: untyped): untyped =
  #when compiles((var t=p; t)): p
  #else:
  procInst(p)
    #var t =
    #t
macro `>>`*(px: tuple, y: any): auto =
  #echo "begin >>:"
  #echo px.treerepr
  #echo "kernel type:"
  #echo px[0].getTypeImpl.treerepr
  #echo "kernel args:"
  #echo y.treerepr
  #var a = y
  #if y.kind != nnkPar: a = newNimNode(nnkPar).addChildrenFrom(y)
  result = newCall(ident("cudaLaunch"))
  let krnl = newCall(px[0]).addChildrenFrom(y)
  #echo "kernel inst call:"
  #echo krnl.treerepr
  result.add getAst(getInst(krnl))[0]
  result.add px[1][0]
  result.add px[1][1]
  for c in y:
    result.add c
  #echo "kernel launch body:"
  #echo result.treerepr

macro cuda*(s,p: untyped): auto =
  #echo "begin cuda:"
  #echo s.treerepr
  let ss = s.strVal
  #echo "proc:"
  #echo p.treerepr
  if p.kind == nnkProcDef:
    result = p
  else:
    result = p[0]
  result.addPragma parseExpr("{.codegenDecl:\""&ss&" $# $#$#\".}")[0]
  result.body = getAst(cudaDefs(result.body))
  #echo "end cuda:"
  #echo result.treerepr
template cudaGlobal*(p: untyped): auto = cuda("__global__",p)


when isMainModule:
  type FltArr = array[0,float32]
  proc vectorAdd*(A: FltArr; B: FltArr; C: var FltArr; n: int32)
    {.cudaGlobal.} =
    var i = blockDim.x * blockIdx.x + threadIdx.x
    if i < n:
      C[i] = A[i] + B[i]

  proc test =
    var n = 50000.cint
    var
      a = newSeq[float32](n)
      b = newSeq[float32](n)
      c = newSeq[float32](n)
    var threadsPerBlock: cint = 256
    var blocksPerGrid: cint = (n + threadsPerBlock - 1) div threadsPerBlock

    cudaLaunch(vectorAdd, blocksPerGrid, threadsPerBlock, a, b, c, n)

  test()
