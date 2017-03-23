type SomeNumber2* = SomeInteger | SomeReal
template `:=`*(x: var SomeNumber, y: SomeNumber2) =
  x = (type(x))(y)
template `+=`*(x: var SomeNumber, y: SomeNumber2) =
  x += (type(x))(y)

type
  Complex*[T] = object
    re*,im*: T
template `:=`*[T](x: var Complex[T], y: SomeNumber) =
  let z = y
  x.re := z
  x.im := 0
template `:=`*[T](x: var Complex[T], y: Complex[T]) =
  x.re = y.re
  x.im = y.im
template `+=`*[T](x: var Complex[T], y: SomeNumber) =
  let z = y
  x.re += z
template `+=`*[T](x: var Complex[T], y: Complex[T]) =
  let z = y
  x.re += z.re
  x.im += z.im
template `+`*[T](x,y: Complex[T]): untyped =
  var r {.noInit.}: Complex[type(x.re+y.re)]
  r.re = x.re + y.re
  r.im = x.im + y.im
  r
template `*`*[T](x,y: Complex[T]): untyped =
  var r {.noInit.}: Complex[type(x.re*y.re)]
  r.re = x.re*y.re - x.im*y.im
  r.im = x.re*y.im + x.im*y.re
  r

type
  Colmat*[T] = object
    d*: array[3,array[3,Complex[T]]]
template `[]`*(x: Colmat, i,j: int): untyped = x.d[i][j]
template `:=`*[T](x: var Colmat[T], y: SomeNumber) =
  let z = y
  for i in 0..<x.d.len:
    for j in 0..<x.d[0].len:
      if i==j:
        x.d[i][j] := z
      else:
        x.d[i][j] := 0
template `:=`*[T](x: var Colmat[T], y: Colmat[T]) =
  let z = y
  for i in 0..<x.d.len:
    for j in 0..<x.d[0].len:
      x.d[i][j] = z.d[i][j]
template `+=`*[T](x: var Colmat[T], y: Colmat[T]) =
  let z = y
  for i in 0..<x.d.len:
    for j in 0..<x.d[0].len:
      x.d[i][j] += z.d[i][j]
template `+`*[T](x,y: Colmat[T]): untyped =
  var r {.noInit.}: Colmat[type(x.d[0][0].re+y.d[0][0].re)]
  for i in 0..<r.d.len:
    for j in 0..<r.d[0].len:
      r.d[i][j] = x.d[i][j] + y.d[i][j]
  r
template `*`*[T](x,y: Colmat[T]): untyped =
  var r {.noInit.}: Colmat[type(x.d[0][0].re*y.d[0][0].re)]
  for i in 0..<r.d.len:
    for j in 0..<r.d[0].len:
      var t = x.d[i][0] * y.d[0][j]
      for k in 1..<y.d.len:
        t += x.d[i][k] * y.d[k][j]
      r.d[i][j] = t
  r

when isMainModule:
  var x,y,z: ref Complex[float]
  x.new
  y.new
  z.new
  x[] += y[]*z[]
  echo x[]
