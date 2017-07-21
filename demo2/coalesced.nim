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
  M = 2                         # Number of RegisterWords in a MemoryWord
  # V = 4                         # Inner array length
  # M = 1                         # Number of RegisterWords in a MemoryWord

type
  Coalesced*[T] = object
    p*: ptr T                   # pointer to an array of T
    n*: int                     # the length of the array being coalesced
  CoalescedObj[T] = object
    o*: Coalesced[T]            # the coalesced array
    i*: int                     # the index to which we asks
  RegisterWord = int32          # Word fits in a register, 4 bytes for current GPU
  # RegisterWord = int64          # Word fits in a register, 4 bytes for current GPU
  Word[L:static[int]] = array[L,RegisterWord]
  MemoryWord = Word[M]          # The granularity of memory transactions.

# Nim doesn't know the size of any struct for sure without the help of a C/C++ compiler.
# So we use a C++ compiler to check if the user has provided a correct size.
# The following C++ code only works with c++11 or later.
{.emit:"""
template <typename ToCheck, std::size_t ProvidedSize, std::size_t RealSize = sizeof(ToCheck)>
void check_size() {static_assert(ProvidedSize == RealSize, "newCoalesced got the wrong size!");}
""".}
proc newCoalesced*[T](p:ptr T, n:int):auto =
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

template arr[T](p:ptr T):auto =
  type A{.unchecked.} = array[0,T]
  cast[ptr A](p)

proc copy[N:static[int]](x:ptr Word[N], y:ptr RegisterWord, n:int) = # n is number of Word[N] in x
  for i in 0..<n:
    for j in 0..<N: arr(x)[i][j] = arr(y)[i*N+j]
proc copy[N:static[int]](x:ptr RegisterWord, y:ptr Word[N], n:int) = # n is number of Word[N] in y
  for i in 0..<n:
    for j in 0..<N: arr(x)[i*N+j] = arr(y)[i][j]

converter fromCoalesced*[T](x:CoalescedObj[T]):T {.noinit.} =
  mixin structSize
  const N = structSize(T) div (M*sizeof(RegisterWord))
  let p = cast[ptr MemoryWord](x.o.p)
  var m {.noinit.}: array[N,MemoryWord]
  for i in 0..<N: m[i] = arr(p)[((x.i div V)*N + i)*V + x.i mod V]
  copy(cast[ptr RegisterWord](result.addr), m[0].addr, N)

proc `:=`*[T,Y](x:CoalescedObj[T], y:Y) =
  when Y is T:
    mixin structSize
    const N = structSize(T) div (M*sizeof(RegisterWord))
    var y = y
    let p = cast[ptr MemoryWord](x.o.p)
    var m {.noinit.}: array[N,MemoryWord]
    copy(m[0].addr, cast[ptr RegisterWord](y.addr), N)
    for i in 0..<N: arr(p)[((x.i div V)*N + i)*V + x.i mod V] = m[i]
  else:
    mixin `:=`
    var ty {.noinit.} :T
    ty := y
    x := ty

proc `*`*[X,Y](x:CoalescedObj[X], y:CoalescedObj[Y]):auto {.noinit.} =
  let
    tx = fromCoalesced(x)
    ty = fromCoalesced(y)
  mixin `*`
  tx * ty

template `+=`*[T,Y](x:CoalescedObj[T], y:Y) = x := fromCoalesced(x) + y

when isMainModule:
  import strutils
  type T = Word[6]
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
  let y = cast[ptr RegisterWord](x[0].addr)
  for i in 0..<x.len*x[0].len:
    if i mod V == 0: s &= "\n"
    s &= " " & align($arr(y)[i],4)
  s &= "}"
  echo s
