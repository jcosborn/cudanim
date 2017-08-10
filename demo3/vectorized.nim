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

const VLEN* {.intdefine.} = 4    ## SIMD vector length
macro defsimd:auto =
  var s,d:NimNode
  result = newstmtlist()
  if VLEN > 1:
    s = ident("SimdS" & $VLEN)
    d = ident("SimdD" & $VLEN)
    result.add( quote do:
      import qexLite/simd
    )
  else:
    s = ident("float32")
    d = ident("float64")
  result.add( quote do:
    type
      SVec* {.inject.} = `s`
      DVec* {.inject.} = `d`
  )
  # echo result.repr
defsimd()

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
    C = x.M*sizeof(RegisterWord)
    N = getSize(x.T) div C
    S = N*x.V*x.M
  mixin vectorType, elementType
  type V = vectorType(x.V, x.T)
  type E = elementType(x.T)
  let ix = x.i.int
  let p = cast[RWA](cast[RWA](x.o.p)[ix*S].addr)
  var r {.noinit.}: V
  let m = cast[RWA](r.addr)
  when sizeof(E) == C:
    # echo "sizeof(E) = C"
    for i in 0..<S: m[i] = p[i]
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

proc `:=`*[V,M:static[int],X,Y](x:VectorizedObj[V,M,X], y:var Y) =
  mixin vectorType, elementType
  type V = vectorType(x.V, x.T)
  when Y is V:
    const
      C = x.M*sizeof(RegisterWord)
      N = getSize(x.T) div C
      S = N*x.V*x.M
    type E = elementType(x.T)
    let ix = x.i.int
    let p = cast[RWA](cast[RWA](x.o.p)[ix*S].addr)
    # The code violates the strict aliasing rule.  GCC tends to optimize it away.
    # To be safe: always pass `-fno-strict-aliasing` to GCC.
    let m = cast[RWA](y.addr)
    # var m {.noinit.}: array[S,RegisterWord]
    # for i in 0..<S: m[i] = cast[RWA](y.addr)[i]
    when sizeof(E) == C:
      # echo "sizeof(E) = C"
      for i in 0..<S: p[i] = m[i]
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
proc `:=`*[V,M:static[int],X,Y](x:VectorizedObj[V,M,X], y:Y) =
  mixin `:=`,vectorType
  type V = vectorType(x.V, x.T)
  var ty {.noinit.}:V
  ty := y
  x := ty

template `+=`*(x:VectorizedObj, y:typed) = x := x[] + y

proc `*`*[VX,MX,VY,MY:static[int],X,Y](x:VectorizedObj[VX,MX,X], y:VectorizedObj[VY,MY,Y]):auto {.noinit.} =
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

proc `+`*(x,y:ShortVector):auto {.noinit.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] + y[i]
  z
proc `-`*(x,y:ShortVector):auto {.noinit.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] - y[i]
  z
proc `*`*(x,y:ShortVector):auto {.noinit.} =
  const V = x.len
  var z {.noinit.}:ShortVector
  for i in 0..<V: z[i] = x[i] * y[i]
  z
proc `+=`*(x:var ShortVector, y:ShortVector) =
  const V = x.len
  for i in 0..<V: x[i] += y[i]
proc `:=`*(x:var ShortVector, y:ShortVector) =
  const V = x.len
  for i in 0..<V: x[i] = y[i]
proc `:=`*(x:var ShortVector, y:SomeNumber) =
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
  template vectorType(V:static[int],t:typedesc[T]):untyped = array[L,ShortVector[V,int32]]
  template vectorType(V:static[int],t:typedesc[S]):untyped = array[2,ShortVector[V,int32]]
  template elementType[N:static[int],T](t:typedesc[array[N,T]]):untyped = T
  #template elementType(t:typedesc[T]):auto = int32
  #template structSize(t:typedesc[U]):int = L*sizeof(int64)
  template vectorType(V:static[int],t:typedesc[U]):untyped = array[L,ShortVector[V,int64]]
  #template elementType(t:typedesc[U]):auto = int64
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
      s &= " " & align($y[i],4)
    s &= "}"
    echo s
  echo "# Check vectorized access"
  test(4,1,T)
  test(4,2,T)
  test(4,2,S)
  test(4,1,U)
  test(4,2,U)