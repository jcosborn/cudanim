import macros

proc isVarType(x: NimNode): bool =
  if x.kind==nnkVarTy: result = true

proc isMagic(x: NimNode): bool =
  #echo x.treerepr
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

proc has(n:NimNode, k:NimNodeKind):bool =
  for c in n:
    if c.kind == k: return true
  return false

proc rebuild(n:NimNode):NimNode =
  # Typed AST has extra information in its nodes.
  # Replacing nodes in a typed AST can break its consistencies,
  # for which the compiler is not well prepared.
  # Here we simply rebuild some offensive nodes from scratch,
  # and force the compiler to rebuild its type information.

  # Note that the compiler currently (v0.17) only retype the AST
  # after a macro returns, so to preserve type information while
  # traversing the AST, call this proc on the result
  #    result = rebuild result
  # just before macro returns.

  # Special node kinds have to be taken care of.
  if n.kind == nnkConv:
    result = newNimNode(nnkCall, n).add(rebuild n[0], rebuild n[1])
  elif n.kind in nnkCallKinds and n[^1].kind == nnkBracket and
       n[^1].len>0 and n[^1].has(nnkHiddenCallConv):
    # special case of varargs
    result = newCall(n[0])
    for i in 1..<n.len-1: result.add rebuild n[i]
    for c in n[^1]:
      if c.kind == nnkHiddenCallConv: result.add rebuild c[1]
      else: result.add rebuild c
  elif n.kind in nnkCallKinds and n[^1].kind == nnkHiddenStdConv and n[^1][1].kind == nnkBracket and
       n[^1][1].len>0 and n[^1][1].has(nnkHiddenCallConv):
    # Deals with varargs
    result = newCall(n[0])
    for i in 1..<n.len-1: result.add rebuild n[i]
    for c in n[^1][1]:
      if c.kind == nnkHiddenCallConv: result.add rebuild c[1]
      else: result.add rebuild c
  elif n.kind == nnkHiddenStdConv:
    # Generic HiddenStdConv
    result = rebuild n[1]
  elif n.kind in {nnkHiddenDeref,nnkHiddenAddr}:
    # Let the compiler do it again.
    result = rebuild n[0]

  # Strip information from other kinds
  else:
    # If something breaks, try adding the offensive node here.
    #of CallNodes + {nnkBracketExpr}:
    #if n.kind in nnkCallKinds + {nnkBracketExpr,nnkBracket,nnkDotExpr}:
    if n.kind in nnkCallKinds + {nnkBracketExpr,nnkBracket,nnkDotExpr}:
      result = newNimNode(n.kind, n)
    # Copy other kinds of node.
    else:
      result = n.copyNimNode
    for c in n:
      result.add rebuild c

proc replace(n,x,y:NimNode):NimNode =
  if n == x:
    result = y.copyNimTree
  else:
    result = n.copyNimNode
    for c in n:
      result.add c.replace(x,y)

proc replaceSym(b,s,r: NimNode): NimNode =
  # same as replace, but uses eqIdent.
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

proc append(x,y:NimNode) =
  for c in y: x.add c

proc regenSym(n:NimNode):NimNode =
  # Only regen nskVar and nskLet symbols.
  proc get(n:NimNode,k:NimNodeKind):NimNode =
    result = newPar()
    if n.kind == k:
      for d in n:
        if d.kind != nnkIdentDefs or d.len<3:
          echo "Internal ERROR: regenSym: can't handle:"
          echo n.treerepr
          quit 1
        for i in 0..<d.len-2:   # Last 2 is type and value.
          if d[i].kind == nnkSym: result.add d[i]
        for c in d[^1]: result.append c.get k
    else:
      for c in n: result.append c.get k
  result = n.copyNimTree
  for x in result.get nnkLetSection:
    let y = genSym(nskLet, $x.symbol)
    result = result.replace(x,y)
  for x in result.get nnkVarSection:
    let y = genSym(nskVar, $x.symbol)
    result = result.replace(x,y)

proc inlineProcsY*(call: NimNode, procImpl: NimNode): NimNode =
  # echo ">>>>>> inlineProcsY"
  # echo "call:\n", call.repr
  # echo "procImpl:\n", procImpl.repr
  template fp: untyped = procImpl[3]  # formal params
  var sl = if fp[0].kind==nnkEmpty: newStmtList()
    else: newNimNode(nnkStmtListExpr)
  for i in 1..<call.len:  # loop over call arguments
    let (sym,typ) = getParam(fp, i)
    if isVarType(typ):
      #let t = genSym(nskVar, "var" & $nInlineLet & "arg")
      #inc nInlineLet
      #template varX(x,y: untyped): untyped =
      #  var x = y
      #let l = getAst(varX(t, call[i]))
      #sl.add l[0]
      #echo "replacing: ", sym.treerepr
      #echo "with: ", call[i].treerepr
      #let t = newNimNode(nnkHiddenAddr).add(call[i][0])
      #procImpl[6] = procImpl[6].replaceSym(sym, call[i][0])
      #procImpl[6] = procImpl[6].replaceSym(sym, t)
      procImpl[6] = procImpl[6].replaceSym(sym, call[i])
    else:
      #let t = genSym(nskLet, "let" & $nInlineLet & "arg")
      #inc nInlineLet
      #template letX(x,y: untyped): untyped =
      #  let x = y
      #let l = getAst(letX(t, call[i]))
      #sl.add l[0]
      #echo "replacing: ", sym.treerepr
      #procImpl[6] = procImpl[6].replaceSym(sym, t)
      procImpl[6] = procImpl[6].replaceSym(sym, call[i])
  # echo "====== procImpl: ",procImpl.len
  # echo procImpl.treerepr
  # echo "^^^^^^"
  if procImpl.len == 7:
    sl.add newBlockStmt(ident($call[0]), procImpl.body)
  elif procImpl.len == 8:
    # echo "TYPEof fp[0]: ",fp[0].lisprepr," ",fp[0].gettype.treerepr
    # echo "TYPEof pi[7]: ",procImpl[7].lisprepr," ",procImpl[7].gettype.treerepr
    template varX(x,t:untyped):untyped =
      var x: t
    #let t = genSym(nskVar, "arg" & $nInlineLet)
    #sl.add procImpl[6]
    #sl.add procImpl[7]
    #sl.add procImpl.body
    var ty:NimNode
    let
      r = procImpl[7]
      z = genSym(nskVar, $r.symbol)
    proc firstAsgn(n:NimNode):NimNode =
      result = newEmptyNode()
      if n.kind == nnkAsgn and n[0] == r:
        result = n[1]
      else:
        for c in n:
          result = firstAsgn c
          if result.kind != nnkEmpty: break
    if fp[0] == bindsym"auto":
      # Return type is auto, so we need to find a way to get the actual type.
      let x = procImpl.body.firstAsgn
      if x.kind == nnkEmpty:
        echo "Internal ERROR: inlineProcsY: cannot determine the type of the proc:"
        echo procImpl.treerepr
        quit 1
      ty = x.gettypeinst
    else:
      ty = r.gettypeinst
    sl.add getAst(varX(z,ty))
    sl.add newBlockStmt(ident($call[0]), procImpl.body.replace(r,z))
    sl.add z
  else:
    echo "Internal ERROR: inlineProcsY: unforeseen length of the proc implementation: ", procImpl.len
    quit 1
  # echo "====== sl"
  # echo sl.repr
  # echo "^^^^^^"
  proc clean(n:NimNode):NimNode =
    proc getFastAsgn(x:NimNode):NimNode =
      result = newPar()
      if x.kind == nnkFastAsgn:
        result.add newPar(x[0],x[1])
      elif x.kind notin AtomicNodes:
        for y in x:
          for c in getFastAsgn(y): result.add c
    proc removeDeclare(n,x:NimNode):NimNode =
      if n.kind == nnkVarSection and n.len == 1 and n[0][0] == x:
        result = newNimNode(nnkDiscardStmt,n).add newStrLitNode(n.lisprepr)
      else:
        result = n.copyNimNode
        for c in n:
          result.add c.removeDeclare x
    result = n.copyNimTree
    let xs = getFastAsgn n
    if xs.len > 0:
      for x in xs:
        let t = genSym(nskVar, $x[0].symbol)
        result = result.removeDeclare(x[0]).replace(x[0],t).replace(x[1],t)
    proc replaceFastAsgn(x:NimNode):NimNode =
      if x.kind == nnkFastAsgn:
        result = newNimNode(nnkDiscardStmt,x).add newStrLitNode(x.lisprepr)
      else:
        result = x.copyNimNode
        if x.kind notin AtomicNodes:
          for c in x:
            result.add c.replaceFastAsgn
    result = result.replaceFastAsgn
  result = regenSym clean sl
  # echo "<<<<<< inlineProcsY"
  # echo result.treerepr

proc callName(x: NimNode): NimNode =
  case x.kind
  of nnkCall,nnkInfix,nnkCommand: result = x[0]
  else:
    quit "callName: unknown kind (" & treeRepr(x) & ")\n" & repr(x)

proc inlineProcsX*(body: NimNode): NimNode =
  # echo ">>>>>> inlineProcsX"
  # echo body.repr
  proc recurse(it: NimNode): NimNode =
    if it.kind in nnkCallKinds and it.callName.kind==nnkSym:
      let procImpl = it.callName.symbol.getImpl
      # echo it.treerepr
      # echo procImpl.repr
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
  # echo "<<<<<< inlineProcsX"
  # echo result.repr

macro inlineProcs*(body: typed): auto =
  # I only function properly inside a proc.  If used at top
  # level, name clash may occur, when inlined procs bring in
  # duplicated names.
  # echo ">>>>>> inlineProcs:"
  #echo body.repr
  # echo body.treerepr
  result = rebuild inlineProcsX(body)
  #result = body
  # echo "<<<<<< inlineProcs:"
  #echo result.repr
  # echo result.treerepr


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

  type T = array[3,float]
  proc loop(x:var T, y:T) =
    let n = 3.0
    for k in 0..<x.len:
      x[k] = n * y[k]
  proc loop2(x:T,y:T):T =
    let n = 0.1
    for k in 0..<x.len:
      result[k] = n * x[k] + y[k]
    for k in 0..<x.len:
      result[k] = n * result[k]
  proc loop3(x:var T,y:T) =
    x.loop y
    x = y.loop2 y
  proc cl =
    var x {.noinit.}: T
    var z {.noinit.}: T
    for i in 0..<x.len: x[i] = i.float
    inlineProcs: z.loop x
    for i in 0..<x.len: echo z[i]
    inlineProcs: z = x.loop2 x
    for i in 0..<x.len: echo z[i]
    inlineProcs:
      z.loop x
      z = x.loop2 x
    for i in 0..<x.len: echo z[i]
    inlineProcs: z.loop3 x
    for i in 0..<x.len: echo z[i]
  cl()
  proc rec =
    var x {.noinit.}: T
    var z {.noinit.}: T
    for i in 0..<x.len: x[i] = i.float
    inlineProcs: z.loop x
    for i in 0..<x.len: echo i," ",z[i]
  inlineProcs:
    rec()
    cl()
