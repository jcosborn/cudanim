import macros

proc isVarType(x: NimNode): bool =
  if x.kind==nnkVarTy: result = true

proc isMagic(x: NimNode): bool =
  #echo x.treerepr
  let pragmas = x[4]
  if pragmas.kind==nnkPragma and pragmas[0].kind==nnkExprColonExpr and
     $pragmas[0][0]=="magic": result = true

proc getParam(fp: NimNode, n: int): auto =
  # n counts from 1
  var n = n-1
  for i in 1..<fp.len:
    let c = fp[i]
    if n >= c.len-2: n -= c.len-2
    else: return (c[n].copyNimTree, c[^2].copyNimTree)

proc has(n:NimNode, k:NimNodeKind):bool =
  for c in n:
    if c.kind == k: return true
  return false

const exprNodes = {
  nnkExprEqExpr,
  nnkExprColonExpr,
  nnkCurlyExpr,
  nnkBracketExpr,
  nnkPragmaExpr,
  nnkDotExpr,
  nnkCheckedFieldExpr,
  nnkDerefExpr,
  nnkIfExpr,
  nnkElifExpr,
  nnkElseExpr,
  nnkStaticExpr,
  nnkStmtListExpr,
  nnkBlockExpr,
  nnkTypeOfExpr }

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
  elif n.kind in nnkCallKinds and n[0] == bindsym"echo" and n.len>0 and n[1].kind == nnkBracket:
    # One dirty hack for the builtin echo, with no nnkHiddenCallConv (this is been caught above)
    result = newCall(n[0])
    for c in n[1]: result.add rebuild c
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
  elif n.kind == nnkHiddenAddr and n[0].kind == nnkHiddenDeref:
    result = rebuild n[0][0]
  elif n.kind == nnkHiddenDeref and n[0].kind == nnkHiddenAddr:
    result = rebuild n[0][0]
  elif n.kind == nnkTypeSection:
    # Type section is special.  Once the type is instantiated, it exists, and we don't want duplicates.
    result = newNimNode(nnkDiscardStmt,n).add(newStrLitNode(n.lisprepr))

  # Strip information from other kinds
  else:
    # If something breaks, try adding the offensive node here.
    #if n.kind in nnkCallKinds + {nnkBracketExpr,nnkBracket,nnkDotExpr}:
    if n.kind in nnkCallKinds + exprNodes + {nnkAsgn}:
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

proc replaceAlt(n,x,y:NimNode, k:NimNodeKind):NimNode =
  # Same as replace but the optional parent node kind k is included in the replacement.
  if n.kind == k and n.len==1 and n[0] == x:
    result = y.copyNimTree
  elif n == x:
    result = y.copyNimTree
  else:
    result = n.copyNimNode
    for c in n:
      result.add c.replaceAlt(x,y,k)

proc replaceNonDeclSym(b,s,r: NimNode, extra:NimNodeKind = nnkEmpty): NimNode =
  # Replace a symbol `s` that's not declared in the body `b` with `r`.
  # Assuming a unique symbol exists.  Only works with trees of symbols.
  var ss: string
  if s.kind==nnkIdent: ss = $s
  else: ss = $(s.symbol)
  # echo "replacing ",ss
  var
    declSyms = newPar()
    theSym = newEmptyNode()
  proc checkSym(n:NimNode) =
    if theSym == n: return
    var f = false
    for c in declSyms:
      if c == n:
        f = true
        break
    if f: return
    elif theSym.kind == nnkEmpty: theSym = n
    else:
      echo "Internal ERROR: replaceNonDeclSym: multiple ",s.repr," found in:"
      echo b.treerepr
      echo "found: ",theSym.lineinfo," :: ",theSym.lisprepr
      echo "found: ",n.lineinfo," :: ",n.lisprepr
      quit 1
  proc find(n:NimNode) =
    # echo "declSyms: ",declSyms.repr
    # echo "theSym: ",theSym.repr
    case n.kind:
    of nnkSym:
      if n.eqIdent ss: checkSym n
    of nnkIdentDefs, nnkConstDef:
      for i in 0..<n.len-2:
        if n[i].eqIdent ss: declSyms.add n[i]
      find n[^1]
    of nnkExprColonExpr:
      find n[1]
    of nnkDotExpr:
      find n[0]
    of nnkProcDef, nnkMethodDef, nnkDo, nnkLambda, nnkIteratorDef,
       nnkTemplateDef, nnkConverterDef:
      echo "Internal ERROR: replaceNonDeclSym: unhandled cases."
      quit 1
    else:
      for c in n: find c
  find b
  # echo "declSyms: ",declSyms.repr
  # echo "theSym: ",theSym.repr
  if theSym.kind != nnkEmpty:
    result = b.replaceAlt(theSym, r, extra)
  else:
    echo "Internal ERROR: replaceNonDeclSym: Couldn't find the symbol ",ss," in body:"
    echo b.treerepr
    quit 1

proc append(x,y:NimNode) =
  for c in y: x.add c

proc regenSym(n:NimNode):NimNode =
  # Only regen nskVar and nskLet symbols.

  # We need to regenerate symbols for multiple inlined procs,
  # because sometimes the compiler still can be confused with
  # "reintroduced symbols".
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
  # echo "call:\n", call.lisprepr
  # echo "procImpl:\n", procImpl.treerepr
  let fp = procImpl[3]  # formal params
  var
    pre = newStmtList()
    body = procImpl.body.copyNimTree
  for i in 1..<call.len:  # loop over call arguments
    # We need to take care of the case when one argument use the same identifier
    # as one formal parameter.  Reusing the formal parameter identifiers is OK.
    let
      (sym,typ) = getParam(fp, i)
      t = genSym(nskLet, $sym)
    template letX(x,y: untyped): untyped =
      let x = y
    # echo "sym: ",sym.lineinfo," :: ",sym.lisprepr
    # echo "typ: ",typ.lineinfo," :: ",typ.lisprepr
    let p = if call[i].kind in {nnkHiddenAddr,nnkHiddenDeref}: call[i][0] else: call[i]
    if isVarType(typ):
      pre.add getAst(letX(t, newNimNode(nnkAddr,p).add p))[0]
      body = body.replaceNonDeclSym(sym, newNimNode(nnkDerefExpr,p).add(t), nnkHiddenDeref)
    else:
      pre.add getAst(letX(t, p))[0]
      body = body.replaceNonDeclSym(sym, t)
  let blockname = genSym(nskLabel, $call[0])
  proc breakReturn(n:NimNode):NimNode =
    if n.kind == nnkReturnStmt:
      result = newStmtList()
      for c in n: result.add c
      result.add newNimNode(nnkBreakStmt, n).add blockname
      if result.len == 1: result = result[0]
    else:
      result = n.copyNimNode
      for c in n: result.add breakReturn c
  body = breakReturn body
  # echo "====== body:"
  # echo body.treerepr
  # echo "^^^^^^"
  var sl:NimNode
  if procImpl.len == 7:
    pre.add body
    sl = newBlockStmt(blockname, pre)
  elif procImpl.len == 8:
    # echo "TYPEof call[0]: ",call[0].lisprepr," ",call[0].gettypeinst.treerepr
    # echo "TYPEof fp[0]: ",fp[0].lisprepr," ",fp[0].gettype.treerepr
    # echo "TYPEof pi[7]: ",procImpl[7].lisprepr," ",procImpl[7].gettype.treerepr
    template varX(x,t:untyped):untyped =
      var x: t
    let
      #ty = call[0].gettypeinst[0][0].gettypeinst
      ty = call[0].gettypeinst[0][0]
      r = procImpl[7]
      z = genSym(nskVar, $r.symbol)
      l = getAst(varX(z,ty))
    # FIXME: check noinit pragma
    pre.add body.replace(r,z)
    # MAYBE: try blockexpr
    sl = newNimNode(nnkStmtListExpr,call).add(l[0], newBlockStmt(blockname, pre), z)
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
    # echo "<<<<<< clean"
    # echo result.treerepr
  result = regenSym clean sl
  #result = clean sl
  # echo "<<<<<< inlineProcsY"
  # echo result.treerepr

proc callName(x: NimNode): NimNode =
  if x.kind in nnkCallKinds: result = x[0]
  else: quit "callName: unknown kind (" & treeRepr(x) & ")\n" & repr(x)

proc inlineProcsX*(body: NimNode): NimNode =
  # echo ">>>>>> inlineProcsX"
  # echo body.repr
  proc recurse(it: NimNode): NimNode =
    if it.kind in nnkCallKinds and it.callName.kind==nnkSym:
      let procImpl = it.callName.symbol.getImpl
      # echo "inspecting call"
      # echo it.repr
      # echo procImpl.repr
      if procImpl.body.kind!=nnkEmpty and
          not isMagic(procImpl) and
          procImpl.kind != nnkIteratorDef:
        result = inlineProcsY(it, procImpl)
        result = recurse(result)
        return
    result = copyNimNode(it)
    for c in it: result.add recurse c
  result = recurse(body)
  # echo "<<<<<< inlineProcsX"
  # echo result.repr

macro inlineProcs*(body: typed): auto =
  # echo ">>>>>> inlineProcs:"
  #echo body.repr
  # echo body.treerepr
  result = rebuild inlineProcsX(body)
  #result = body
  # echo "<<<<<< inlineProcs:"
  # echo result.repr
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

  echo "* Basics"
  a1(1.0)
  a2(1.0)

  echo "* multiple iterators"
  type T = array[3,float]
  proc loop(x:var T, y:T) =
    echo "loop"
    let n = 3.0
    for k in 0..<x.len:
      x[k] = n * y[k]
  proc loop2(x:T,y:T):T =
    echo "loop2"
    let n = 0.1
    for k in 0..<x.len:
      result[k] = n * x[k] + y[k]
    for k in 0..<x.len:
      result[k] = n * result[k]
  proc loop3(x:var T,y:T) =
    echo "loop3"
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

  echo "* top level inlineProcs calling proc with inlineProcs"
  proc rec =
    var x {.noinit.}: T
    var z {.noinit.}: T
    for i in 0..<x.len: x[i] = i.float
    inlineProcs: z.loop x
    for i in 0..<x.len: echo i," ",z[i]
  inlineProcs:
    rec()
    cl()

  echo "* avoid duplicate computations"
  proc inplace(x:var float, y:float) =
    x = x + y
    x = x + 1000*y
  inlineProcs:
    var x {.noinit.}: T
    var z {.noinit.}: T
    for i in 0..<x.len: x[i] = i.float
    z.loop(x.loop2 x)
    for i in 0..<x.len:
      z[i].inplace(0.1*i.float)
      echo i," ",z[i]
    when not defined cpp:       # github issue #4048, cpp breaks mitems
      var s = 0.0
      for m in mitems(z):
        s += m
        m.inplace(1000 * s)
      for i in 0..<x.len: echo z[i]

  echo "* redeclaration of formal params"
  proc redecl(x:var float, y:float) =
    block:
      var x = x
      x += y
      var y = y
      y += 1
      echo x," ",y
    x += 3
    let x = x
    var y = y
    y += x
    echo x," ",y
  block:
    echo "Without inlining:"
    var x = 1.0
    var y = 0.1
    x.redecl(y+0.01)
    echo x," ",y
  block:
    inlineProcs:
      echo "With inlining:"
      var x = 1.0
      var y = 0.1
      x.redecl(y+0.01)
      echo x," ",y

  echo "* Generic parameters"
  proc g[T;N:static[int]](x:array[N,T]) =
    var s = ""
    for i in 0..<N:
      if i>0: s &= " , "
      s &= $x[i]
    echo "x = [ ",s," ] has size ",N*sizeof(T)
  block:
    inlineProcs:
      var v = [0,1,2,3]
      g v

  echo "* object construction"
  proc oc(x:int):auto =
    type A = object
      x:int
    return A(x:x)
  block:
    inlineProcs:
      var x = 3
      echo oc(x).x
