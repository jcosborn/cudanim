import qexLite/metaUtils
import ../inline

type SomeNumber2* = SomeInteger | SomeReal
template `:=`*(x: var SomeNumber, y: SomeNumber2) =
  type tx = type(x)
  x = (tx)(y)
template `+=`*(x: var SomeNumber, y: SomeNumber2) =
  bind `+=`    # So the following += doesn't call this template again.
  type tx = type(x)
  x += (tx)(y)

type
  Complex*[T] = object
    re*,im*: T
proc structSize*[T](t:typedesc[Complex[T]]):int = 2*sizeof(T)
template `:=`*[T](x: var Complex[T], y: T) =
  let z = y
  x.re := z
  x.im := 0
template `:=`*[T](x: var Complex[T], y: SomeNumber) =
  let z = y
  x.re := z
  x.im := 0
template `:=`*[T](x: var Complex[T], y: Complex[T]) =
  let z = y
  x.re := z.re
  x.im := z.im
template `+=`*[T](x: var Complex[T], y: T) =
  let z = y
  x.re += z
template `+=`*[T](x: var Complex[T], y: SomeNumber) =
  let z = y
  x.re += z
template `+=`*[T](x: var Complex[T], y: Complex[T]) =
  let z = y
  x.re += z.re
  x.im += z.im
template `+`*[T](x: Complex[T], y:T): untyped =
  type tx = type(x)
  var r {.noInit.}: tx #Complex[x.T] #Complex[type(x.re+y)]
  r.re := x.re + y
  r.im := x.im
  r
template `+`*[T](x: Complex[T], y:SomeNumber): untyped =
  type tx = type(x)
  var r {.noInit.}: tx #Complex[x.T] #Complex[type(x.re+y)]
  r.re := x.re + y
  r.im := x.im
  r
template `+`*[T](x,y: Complex[T]): untyped =
  type tx = type(x)
  var r {.noInit.}: tx #Complex[x.T] #Complex[type(x.re+y.re)]
  r.re := x.re + y.re
  r.im := x.im + y.im
  r
template `*`*[T](x,y: Complex[T]): untyped =
  type tx = type(x)
  var r {.noInit.}: tx #Complex[x.T] #Complex[type(x.re*y.re)]
  r.re := x.re*y.re - x.im*y.im
  r.im := x.re*y.im + x.im*y.re
  r
template `*=`*[T](x: var Complex[T], y: SomeNumber) =
  let z = y
  x.re *= z
  x.im *= z
template norm2*(xx:Complex):untyped =
  let x = xx
  mixin norm2
  x.re.norm2 + x.im.norm2

type
  Colmat*[N:static[int],T] = object
    d*: array[N,array[N,Complex[T]]]
proc structSize*[N:static[int],T](t:typedesc[Colmat[N,T]]):int = 2*N*N*sizeof(T)
template `[]`*(x: Colmat, i,j: int): untyped = x.d[i][j]
template `:=`*[N:static[int],T](x: var Colmat[N,T], y: SomeNumber) =
  let z = y
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      when i==j:
        x.d[i][j] := z
      else:
        x.d[i][j] := 0
template `:=`*[N:static[int],T](x: var Colmat[N,T], y: Colmat[N,T]) =
  let z = y
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      x.d[i][j] := z.d[i][j]
template `+=`*[N:static[int],T](x: var Colmat[N,T], y: Colmat[N,T]) =
  let z = y
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      x.d[i][j] += z.d[i][j]
template `+`*[N:static[int],T](x,y: Colmat[N,T]): untyped =
  let xx = x
  let yy = y
  var r {.noInit.}: Colmat[N,type(x.d[0][0].re)]
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      r.d[i][j] := xx.d[i][j] + yy.d[i][j]
  r
template `*`*[N:static[int],T](x,y: Colmat[N,T]): untyped =
  let xx = x
  let yy = y
  var r {.noInit.}: Colmat[N,type(x.d[0][0].re)]
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      r.d[i][j] := xx.d[i][0] * yy.d[0][j]
    staticfor k, 1, N-1:
      staticfor j, 0, N-1:
        r.d[i][j] += xx.d[i][k] * yy.d[k][j]
  r
template `*=`*[N:static[int],T](x: var Colmat[N,T], y: SomeNumber) =
  let z = y
  staticfor i, 0, N-1:
    staticfor j, 0, N-1:
      x.d[i][j] *= z
template norm2*(xx:Colmat):untyped =
  let x = xx
  var r {.noinit.}: type(x.d[0][0].re.norm2)
  const n = x.N-1
  mixin norm2
  r = x.d[0][0].norm2
  staticfor i, 0, n:
    staticfor j, 0, n:
      when (i != 0) and (j != 0):
        r += x.d[i][j].norm2
  r

when isMainModule:
  var x,y,z: ref Complex[float]
  x.new
  y.new
  z.new
  x[] += y[]*z[]
  echo x[]
