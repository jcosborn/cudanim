import macros

proc isVarType(x: NimNode): bool =
  if x.kind==nnkVarTy: result = true

proc isMagic(x: NimNode): bool =
  let pragmas = x[4]
  if pragmas.kind==nnkPragma and pragmas[0].kind==nnkExprColonExpr and
     $pragmas[0][0]=="magic": result = true

proc getParam(fp: NimNode, n: int): auto =
  var
    sym,typ: NimNode
    c = 1
    i = 1
  while i<fp.len:
    var j=0
    while j<fp[i].len-2:
      if c==n:
        sym = fp[i][j]
        typ = fp[i][^2]
        result = (sym,typ)
      inc c
      inc j
    inc i

proc replaceSym(b,s,r: NimNode): NimNode =
  var ss: string
  if s.kind==nnkIdent: ss = $s
  else: ss = $(s.symbol)
  proc recurse(it: NimNode): NimNode =
    if eqIdent(it, ss):
      return copyNimTree(r)
    result = copyNimNode(it)
    for i in 0..<it.len:
      result.add recurse(it[i])
  result = recurse(b)

var nInlineLet{.compileTime.} = 0

proc inlineProcsY*(call: NimNode, procImpl: NimNode): NimNode =
  template fp: untyped = procImpl[3]  # formal params
  var sl = if fp[0].kind==nnkEmpty: newStmtList()
    else: newNimNode(nnkStmtListExpr)
  for i in 1..<call.len:  # loop over call arguments
    let (sym,typ) = getParam(fp, i)
    if isVarType(typ):
      echo "replacing: ", sym.treerepr
      echo "with: ", call[i].treerepr
      #let t = newNimNode(nnkHiddenAddr).add(call[i][0])
      #procImpl[6] = procImpl[6].replaceSym(sym, call[i][0])
      #procImpl[6] = procImpl[6].replaceSym(sym, t)
      procImpl[6] = procImpl[6].replaceSym(sym, call[i])
    else:
      #let t = genSym(nskLet, "arg" & $nInlineLet)
      #inc nInlineLet
      #template letX(x,y: untyped): untyped =
      #  let x = y
      #let l = getAst(letX(t, call[i]))
      #sl.add l[0]
      #echo "replacing: ", sym.treerepr
      #procImpl[6] = procImpl[6].replaceSym(sym, t)
      procImpl[6] = procImpl[6].replaceSym(sym, call[i])
  if procImpl.len<=7:
    sl.add procImpl[6]
  else:
    #echo fp[0].gettype.gettype.treerepr
    #echo procImpl[7].treerepr
    #echo procImpl[7].gettype.treerepr
    #template varX(x,t:untyped):untyped =
    #  var x: t
    #let t = genSym(nskVar, "arg" & $nInlineLet)
    #sl.add getAst(varX(t,procImpl[7].gettype))
    #sl.add procImpl[6]
    #sl.add procImpl[7]
    sl.add procImpl.body[1]
    #sl.add procImpl[7]
  return sl

proc callName(x: NimNode): NimNode =
  case x.kind
  of nnkCall,nnkInfix: result = x[0]
  else:
    quit "callName: unknown kind (" & treeRepr(x) & ")\n" & repr(x)

proc inlineProcsX*(body: NimNode): NimNode =
  proc recurse(it: NimNode): NimNode =
    if it.kind in nnkCallKinds and it.callName.kind==nnkSym:
      let procImpl = it.callName.symbol.getImpl
      echo it.treerepr
      echo procImpl.treerepr
      echo procImpl[1].treerepr
      if procImpl.body.kind!=nnkEmpty and
          not isMagic(procImpl) and
          procImpl.kind != nnkIteratorDef:
        result = inlineProcsY(it, procImpl)
        result = recurse(result)
        return
    result = copyNimNode(it)
    for i in 0..<it.len:
      result.add recurse(it[i])
  result = recurse(body)

macro inlineProcs*(body: typed): auto =
  echo "begin inlineProcs:"
  echo body.repr
  echo body.treerepr
  result = inlineProcsX(body)
  echo "end inlineProcs:"
  echo result.repr
  echo result.treerepr


when isMainModule:
  proc f1(r: var any; x: any) = r = 2*x
  proc f2(x: any): auto = 2*x

  proc a1(x: float) =
    inlineProcs:
      var r: float
      var s: type(r)
      f1(r, x)
  proc a2(x: float) =
    inlineProcs:
      var r = f2(x)

  a1(1.0)
  a2(1.0)
