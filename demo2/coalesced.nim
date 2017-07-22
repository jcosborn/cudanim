#[

Following Kate's idea of coalesced_ptr in C++, we use a wrapper
object type to hide the actual coalesced memory layout here.
Original comments from Kate's coalesced_ptr.h follows:

  A smart pointer that automatically provide coalesced memory
  transcations for arrays of arbtrary structures.  Given a structure
  T, of size S bytes, e.g.,

  struct T {
    char a[S];
  }

  in an array with sites elements

  T t[sites];

  using a coalesced_ptr will split the structure for reading and
  writing to memory as an array of structures of array of structures (AoSoAoS),
  where:
    - the inner structure size is given by memory_word_size
    - the inner array size is given by site_vector
    - the outer structure size is given by sizeof(T)/memory_word_size
    - the outer array size is given by sites/site_vector

]#

import macros

const
  V = 32                        # Inner array length
  M = 2                         # Number of RegisterWords in a MemoryWord, which is the granularity of memory transactions.
  # M = 1                         # Number of RegisterWords in a MemoryWord, which is the granularity of memory transactions.

type
  Coalesced*[T] = object
    p*: ptr T                   # pointer to an array of T
    n*: int                     # the length of the array being coalesced
  CoalescedObj[T] = object
    o*: Coalesced[T]            # the coalesced array
    i*: int                     # the index to which we asks
  RegisterWord = int32          # Word fits in a register, 4 bytes for current GPU
when M == 1:
  type
    MemoryWord = object
      a*:RegisterWord
elif M == 2:
  type
    MemoryWord = object
      a*,b*:RegisterWord
elif M == 4:
  type
    MemoryWord = object
      a*,b*,c*,d*:RegisterWord
else:
  {.fatal:"Unsupported memory size " & $M.}

# Nim doesn't know the size of any struct for sure without the help of a C/C++ compiler.
# So we use a C++ compiler to check if the user has provided a correct size.
# The following C++ code only works with c++11 or later.
{.emit:"""
template <typename ToCheck, std::size_t ProvidedSize, std::size_t RealSize = sizeof(ToCheck)>
void check_size() {static_assert(ProvidedSize == RealSize, "newCoalesced got the wrong size!");}
""".}
proc newCoalesced*[T](p:ptr T, n:int):auto {.noinit.} =
  when compiles((const size = sizeof(T))):
    const size = sizeof(T)
  else:
    mixin structSize
    const size = structSize(T)
  {.emit:"check_size<`T`, `size`>();".}
  const N = size div (M*sizeof(RegisterWord))
  when N*(M*sizeof(RegisterWord)) != size: {.fatal:"sizeof(T) must be divisible by memory word size."}
  if n mod V != 0:
    echo "Array length for Coalesced must be multiples of ",V
    quit 1
  Coalesced[T](p:p, n:n)
proc `[]`*[T](x:Coalesced[T], i:int):auto = CoalescedObj[T](o:x, i:i)
proc len*[T](x:Coalesced[T]):auto = x.n

type
  RWA {.unchecked.} = array[0,RegisterWord]
  MWA {.unchecked.} = array[0,MemoryWord]

proc copy(x:ptr MemoryWord, y:ptr RegisterWord, n:int) = # n is number of MemoryWord in x
  let
    x = cast[ptr MWA](x)
    y = cast[ptr RWA](y)
  for i in 0..<n:
    x[i].a = y[M*i]
    when M > 1:
      x[i].b = y[M*i+1]
    when M > 2:
      x[i].c = y[M*i+2]
      x[i].d = y[M*i+3]
proc copy(x:ptr RegisterWord, y:ptr MemoryWord, n:int) = # n is number of MemoryWord in y
  let
    x = cast[ptr RWA](x)
    y = cast[ptr MWA](y)
  for i in 0..<n:
    x[M*i] = y[i].a
    when M > 1:
      x[M*i+1] = y[i].b
    when M > 2:
      x[M*i+2] = y[i].c
      x[M*i+3] = y[i].d

converter fromCoalesced*[T](x:CoalescedObj[T]):T {.noinit.} =
  mixin structSize
  const N = structSize(T) div (M*sizeof(RegisterWord))
  let p = cast[ptr MWA](x.o.p)
  var m {.noinit.}: array[N,MemoryWord]
  for i in 0..<N: m[i] = p[((x.i div V)*N + i)*V + x.i mod V]
  copy(cast[ptr RegisterWord](result.addr), m[0].addr, N)

proc `:=`*[T,Y](x:CoalescedObj[T], y:Y) =
  when Y is T:
    mixin structSize
    const N = structSize(T) div (M*sizeof(RegisterWord))
    var y {.noinit.} = y
    let p = cast[ptr MWA](x.o.p)
    var m {.noinit.}: array[N,MemoryWord]
    copy(m[0].addr, cast[ptr RegisterWord](y.addr), N)
    for i in 0..<N: p[((x.i div V)*N + i)*V + x.i mod V] = m[i]
  else:
    mixin `:=`
    var ty {.noinit.} :T
    ty := y
    x := ty

proc `*`*[X,Y](x:CoalescedObj[X], y:CoalescedObj[Y]):auto {.noinit.} =
  let
    tx {.noinit.} = fromCoalesced(x)
    ty {.noinit.} = fromCoalesced(y)
  mixin `*`
  tx * ty

template `+=`*[T,Y](x:CoalescedObj[T], y:Y) = x := fromCoalesced(x) + y

when isMainModule:
  import strutils
  type T = array[6,RegisterWord]
  proc structSize(t:typedesc[T]):int = 24
  var x {.noinit.}: array[64,T]
  let p = newCoalesced(x[0].addr, x.len)
  for i in 0..<p.len:
    var t {.noinit.}: T
    for j in 0..<t.len: t[j] = RegisterWord(100*i + j)
    p[i] := t
  var s:string
  s = "Lexical order: p = {"
  for i in 0..<p.len:
    let t:T = p[i]
    s &= "\n["
    for j in 0..<t.len: s &= " " & align($t[j],4)
    s &= " ]"
  s &= "}"
  echo s
  s = "Memory layout: x = {"
  let y = cast[ptr RWA](x[0].addr)
  for i in 0..<x.len*x[0].len:
    if i mod V == 0: s &= "\n"
    s &= " " & align($y[i],4)
  s &= "}"
  echo s
