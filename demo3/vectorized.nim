#[

Use the same memory layout as Coalesced, but stricter.  It only
works for array objects of a single type.

For example, to wrap a pointer and reorganize the memory layout
of an array of Obj[T], given
  p: ptr Obj[T]
where Obj[T] must be a homogeneous array of T,
we can use
  p: Coalesced[Obj[T]]
so that when indexed for ShortVector
  p[i]
where i is of type ShortVectorIndex[T]
returns an object of type
  VectorizedObj[Obj,T]
which will be converted on demand, and behave as a
  Obj[ShortVector[T]]

Note that efficient operations of ShortVector[T] should be
defined corresponding to those of T.

]#

import coalesced
import macros

const CPUVLEN* {.intdefine.} = 256 ## CPU SIMD vector length in bits.  Off if zero.
const SupportedCPUVLENs = {128,256,512}
const oneByte = 8
macro defsimd:auto =
  var s,d:NimNode
  var
    ss = newIntLitNode(4)
    ds = newIntLitNode(8)
  result = newstmtlist()
  if CPUVLEN == 0:
    s = ident("float32")
    d = ident("float64")
  elif CPUVLEN in SupportedCPUVLENs:
    const
      sl = CPUVLEN div (oneByte*sizeof(float32))
      dl = CPUVLEN div (oneByte*sizeof(float64))
    s = ident("SimdS" & $sl)
    d = ident("SimdD" & $dl)
    ss = newIntLitNode(CPUVLEN div oneByte)
    ds = ss
    result.add( quote do:
      import qexLite/simd
    )
  else:
    echo "ERROR: unsupported value of CPUVLEN: ", CPUVLEN
    quit 1
  result.add( quote do:
    type
      SVec* {.inject.} = `s`
      DVec* {.inject.} = `d`
    template structSize*(t:typedesc[SVec]):int = `ss`
    template structSize*(t:typedesc[DVec]):int = `ds`
  )
  # echo result.repr
defsimd()
template vectorizedElementType*(t:typedesc):untyped =
  when t is float32: SVec
  elif t is float64: DVec
  else: t
template vectorType(vlen:static[int],t:typedesc):untyped =
  mixin elementType,vectorType
  type E = elementType(t)
  type VE = vectorizedElementType(E)
  const
    mvlen = getsize(VE) div sizeof(E) # guaranteed to be divisible
    svlen = vlen div mvlen
  when svlen*mvlen != vlen:
    {.fatal:"Inner vector length " & $vlen & " not divisible by machine vector length " & $mvlen.}
  type SV = ShortVector[svlen,VE]
  vectorType(t,SV)

type
  ShortVector*[V:static[int],E] = object
    a*:array[V,E]
  ShortVectorIndex* = distinct int
  VectorizedObj*[V,M:static[int],T] = object
    o*:Coalesced[V,M,T]
    i*:ShortVectorIndex

template `[]`*(x:Coalesced, ix:ShortVectorIndex):untyped = VectorizedObj[x.V,x.M,x.T](o:x,i:ix)
template veclen*(x:Coalesced):untyped = x.n div x.V

template `[]`*(x:ShortVector, i:int):untyped = x.a[i]
template `[]=`*(x:var ShortVector, i:int, y:typed) = x.a[i] = y
template len*(x:ShortVector):int = x.V

type RWA{.unchecked.} = ptr array[0,RegisterWord]

template fromVectorized*(x:VectorizedObj):untyped =
  const
    C = x.M*sizeof(RegisterWord) # MemoryWord size
    N = getSize(x.T) div C       # Number of MemoryWord in the type x.T
    S = N*x.V*x.M                # Number of RegisterWord in a block of x.V objects
  mixin vectorType, elementType
  type
    E = elementType(x.T)
    VE = vectorizedElementType(E) # Machine simd vector if available
    VEA{.unchecked.} = ptr array[0,VE]
    V = vectorType(x.V,x.T)
  const VL = (x.V * getSize(x.T)) div getSize(VE) # Number of vectorized element in a block of x.V objects
  let ix = x.i.int
  var r {.noinit.}: V
  let
    p = cast[RWA](cast[RWA](x.o.p)[ix*S].addr)
    vp = cast[VEA](cast[RWA](x.o.p)[ix*S].addr)
    m = cast[RWA](r.addr)
    vm = cast[VEA](r.addr)
  when sizeof(E) == C:
    # echo "sizeof(E) = C"
    # for i in 0..<S: m[i] = p[i]
    for i in 0..<VL: vm[i] = vp[i]
  elif sizeof(E) > C:
    # echo "sizeof(E) > C"
    const L = sizeof(E) div C
    when L*C != sizeof(E):
      # We can deal with this but let's leave it for future exercises.
      {.fatal:"Vector element size not divisible by memory word size.".}
    for i in 0..<N:
      for j in 0..<x.V:
        for k in 0..<x.M:
          m[x.V*x.M*L*(i div L) + x.M*L*j + k + x.M*(i mod L)] = p[x.V*x.M*i + x.M*j + k]
  elif sizeof(E) >= sizeof(RegisterWord): # sizeof(E) < C
    # echo "sizeof(RegisterWord) <= sizeof(E) < C"
    const
      L = C div sizeof(E)
      K = sizeof(E) div sizeof(RegisterWord)
    # x.M = L*K
    when K*sizeof(RegisterWord) != sizeof(E):
      # We can deal with this but let's leave it for future exercises.
      {.fatal:"Vector element size not divisible by register word size.".}
    when L*sizeof(E) != C or K*sizeof(RegisterWord) != sizeof(E):
      # We can deal with this but let's leave it for future exercises.
      {.fatal:"Memory word size not divisible by vector element size.".}
    for i in 0..<N:
      for j in 0..<x.V:
        for k in 0..<x.M:
          m[x.V*K*(k div K) + x.V*x.M*i + K*j + (k mod K)] = p[x.V*x.M*i + x.M*j + k]
  else:
    # We can deal with this but let's leave it for future exercises.
    {.fatal:"Register word size larger than vector element size.".}
  r
macro `[]`*(x:VectorizedObj, ys:varargs[untyped]):untyped =
  let o = newCall(bindsym"fromVectorized", x)
  if ys.len == 0:
    result = o
  else:
    result = newCall("[]", o)
    for y in ys: result.add y

proc `:=`*[V,M:static[int],X,Y](x:VectorizedObj[V,M,X], y:var Y) {.inline.} =
  mixin vectorType, elementType
  type E = elementType(x.T)
  type V = vectorType(x.V,x.T)
  when Y is V:
    const
      C = x.M*sizeof(RegisterWord)
      N = getSize(x.T) div C
      S = N*x.V*x.M
    let ix = x.i.int
    type
      VE = vectorizedElementType(E)
      VEA{.unchecked.} = ptr array[0,VE]
    const VL = (x.V * getSize(x.T)) div getSize(VE)
    let
      p = cast[RWA](cast[RWA](x.o.p)[ix*S].addr)
      vp = cast[VEA](cast[RWA](x.o.p)[ix*S].addr)
      m = cast[RWA](y.addr)
      vm = cast[VEA](y.addr)
    when sizeof(E) == C:
      # echo "sizeof(E) = C"
      # for i in 0..<S: p[i] = m[i]
      for i in 0..<VL: vp[i] = vm[i]
    elif sizeof(E) > C:
      # echo "sizeof(E) > C"
      const L = sizeof(E) div C
      when L*C != sizeof(E):
        # We can deal with this but let's leave it for future exercises.
        {.fatal:"Vector element size not divisible by memory word size.".}
      for i in 0..<N:
        for j in 0..<x.V:
          for k in 0..<x.M:
            p[x.V*x.M*i + x.M*j + k] = m[x.V*x.M*L*(i div L) + x.M*L*j + k + x.M*(i mod L)]
    elif sizeof(E) >= sizeof(RegisterWord): # sizeof(E) < C
      # echo "sizeof(RegisterWord) <= sizeof(E) < C"
      const
        L = C div sizeof(E)
        K = sizeof(E) div sizeof(RegisterWord)
      # x.M = L*K
      when K*sizeof(RegisterWord) != sizeof(E):
        # We can deal with this but let's leave it for future exercises.
        {.fatal:"Vector element size not divisible by register word size.".}
      when L*sizeof(E) != C or K*sizeof(RegisterWord) != sizeof(E):
        # We can deal with this but let's leave it for future exercises.
        {.fatal:"Memory word size not divisible by vector element size.".}
      for i in 0..<N:
        for j in 0..<x.V:
          for k in 0..<x.M:
            p[x.V*x.M*i + x.M*j + k] = m[x.V*K*(k div K) + x.V*x.M*i + K*j + (k mod K)]
    else:
      # We can deal with this but let's leave it for future exercises.
      {.fatal:"Register word size larger than vector element size.".}
  else:
    var ty {.noinit.}:V
    ty := y
    x := y
proc `:=`*[V,M:static[int],X,Y](x:VectorizedObj[V,M,X], y:Y) {.inline.} =
  mixin `:=`,vectorType,elementType
  type V = vectorType(x.V,x.T)
  var ty {.noinit.}:V
  ty := y
  x := ty

template `+=`*(xx:VectorizedObj, yy:typed) =
  let
    x = xx
    y = yy
  x := x[] + y

proc `*`*[VX,MX,VY,MY:static[int],X,Y](x:VectorizedObj[VX,MX,X], y:VectorizedObj[VY,MY,Y]):auto {.noinit,inline.} =
  let
    tx {.noinit.} = x[]
    ty {.noinit.} = y[]
  mixin `*`
  tx * ty

iterator vectorIndices*(x:Coalesced):auto =
  var i = 0
  while i < x.veclen:
    yield ShortVectorIndex(i)
    inc i

proc `+`*(x:ShortVector, y:SomeNumber):auto {.noinit,inline.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] + y
  z
proc `+`*(x,y:ShortVector):auto {.noinit,inline.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] + y[i]
  z
proc `-`*(x,y:ShortVector):auto {.noinit,inline.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] - y[i]
  z
proc `*`*(x,y:ShortVector):auto {.noinit,inline.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] * y[i]
  z
proc `+=`*(x:var ShortVector, y:ShortVector) {.inline.} =
  const V = x.len
  for i in 0..<V: x[i] += y[i]
proc `:=`*(x:var ShortVector, y:ShortVector) {.inline.} =
  const V = x.len
  for i in 0..<V: x[i] = y[i]
proc `:=`*(x:var ShortVector, y:SomeNumber) {.inline.} =
  const V = x.len
  for i in 0..<V: x[i] := y

when isMainModule:
  import strutils, typetraits
  const L = 6
  type
    T = array[L,int32]
    S = array[2,int32]
    U = array[L,int64]
  template structSize[N:static[int],T](t:typedesc[array[N,T]]):int = N*sizeof(T)
  template elementType[N:static[int],T](t:typedesc[array[N,T]]):untyped = T
  template vectorType(t:typedesc[T],v:typedesc):untyped = array[L,v]
  template vectorType(t:typedesc[S],v:typedesc):untyped = array[2,v]
  template vectorType(t:typedesc[U],v:typedesc):untyped = array[L,v]
  proc test(v,m:static[int],ty:typedesc) =
    echo "### TEST ",v," ",m," ",ty.name
    var x {.noinit.}:array[12,ty]
    let p = newCoalesced(v,m,x[0].addr,x.len)
    # for i in 0..<p.len:         # CoalescedObj access
    #   var t {.noinit.}: ty
    #   for j in 0..<t.len: t[j] = type(t[j])(100*i + j)
    #   p[i] := t
    for i in vectorIndices(p):  # VectorizedObj asignment
      var t {.noinit.}: vectorType(v,ty)
      for j in 0..<t.len:
        for k in 0..<v:
          t[j][k] = type(t[j][k])(100*(v*i.int+k) + j)
      p[i] := t
    var s:string
    s = "Lexical order: p = {"
    for i in vectorIndices(p):  # VectorizedObj access
      let t = p[i][]
      s &= "\n["
      for k in 0..<t[0].len:    # Inner vector loop
        for j in 0..<t.len: s &= " " & align($t[j][k],4)
      s &= " ]"
    s &= "}"
    echo s
    s = "Memory layout: x = {"
    let y = cast[RWA](x[0].addr)
    for i in 0..<(sizeof(elementType(ty)) div sizeof(RegisterWord))*x.len*x[0].len:
      if i mod (p.V*p.M) == 0: s &= "\n"
      s &= " " & align($cast[uint32](y[i]),4)
    s &= "}"
    echo s
  echo "# Check vectorized access"
  test(4,1,T)
  test(4,2,T)
  test(4,2,S)
  test(4,1,U)
  test(4,2,U)
