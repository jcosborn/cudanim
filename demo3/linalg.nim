import ../inline

type SomeNumber2* = SomeInteger | SomeReal
template `:=`*(x: var SomeNumber, y: SomeNumber2) =
  x = (type(x))(y)
template `+=`*(x: var SomeNumber, y: SomeNumber2) =
  bind `+=`    # So the following += doesn't call this template again.
  x += (type(x))(y)

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
  var r {.noInit.}: Complex[type(x.re+y)]
  r.re := x.re + y
  r.im := x.im
  r
template `+`*[T](x: Complex[T], y:SomeNumber): untyped =
  var r {.noInit.}: Complex[type(x.re+y)]
  r.re := x.re + y
  r.im := x.im
  r
template `+`*[T](x,y: Complex[T]): untyped =
  var r {.noInit.}: Complex[type(x.re+y.re)]
  r.re := x.re + y.re
  r.im := x.im + y.im
  r
template `*`*[T](x,y: Complex[T]): untyped =
  var r {.noInit.}: Complex[type(x.re*y.re)]
  r.re := x.re*y.re - x.im*y.im
  r.im := x.re*y.im + x.im*y.re
  r

type
  Colmat*[N:static[int],T] = object
    d*: array[N,array[N,Complex[T]]]
proc structSize*[N:static[int],T](t:typedesc[Colmat[N,T]]):int = 2*N*N*sizeof(T)
template `[]`*(x: Colmat, i,j: int): untyped = x.d[i][j]
template `:=`*[N:static[int],T](x: var Colmat[N,T], y: SomeNumber) =
  let z = y
  for i in 0..<N:
    for j in 0..<N:
      if i==j:
        inlineProcs:
          x.d[i][j] := z
      else:
        inlineProcs:
          x.d[i][j] := 0
template `:=`*[N:static[int],T](x: var Colmat[N,T], y: Colmat[N,T]) =
  let z = y
  for i in 0..<N:
    for j in 0..<N:
      inlineProcs:
        x.d[i][j] := z.d[i][j]
template `+=`*[N:static[int],T](x: var Colmat[N,T], y: Colmat[N,T]) =
  let z = y
  for i in 0..<N:
    for j in 0..<N:
      inlineProcs:
        x.d[i][j] += z.d[i][j]
template `+`*[N:static[int],T](x,y: Colmat[N,T]): untyped =
  let xx = x
  let yy = y
  var r {.noInit.}: Colmat[N,type(xx.d[0][0].re+yy.d[0][0].re)]
  for i in 0..<N:
    for j in 0..<N:
      inlineProcs:
        r.d[i][j] := xx.d[i][j] + yy.d[i][j]
  r
template `*`*[N:static[int],T](x,y: Colmat[N,T]): untyped =
  let xx = x
  let yy = y
  var r {.noInit.}: Colmat[N,type(xx.d[0][0].re*yy.d[0][0].re)]
  for i in 0..<N:
    for j in 0..<N:
      inlineProcs:
        r.d[i][j] := xx.d[i][0] * yy.d[0][j]
    for k in 1..<N:
      for j in 0..<N:
        inlineProcs:
          r.d[i][j] += xx.d[i][k] * yy.d[k][j]
  r

when isMainModule:
  var x,y,z: ref Complex[float]
  x.new
  y.new
  z.new
  x[] += y[]*z[]
  echo x[]
