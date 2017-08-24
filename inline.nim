import macros

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

proc replaceId(n,x,y:NimNode):NimNode =
  # Same as replace but only replace eqIdent identifier.
  if n.kind == nnkIdent and n.eqIdent($x):
    result = y.copyNimTree
  else:
    result = n.copyNimNode
    for c in n:
      result.add c.replaceId(x,y)

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

proc replaceExcl(n,x,y:NimNode, k:NimNodeKind):NimNode =
  # Same as replace but the optional parent node kind k excluds the replacement.
  if n.kind == k and n.len==1 and n[0] == x:
    result = n.copyNimTree
  elif n == x:
    result = y.copyNimTree
  else:
    result = n.copyNimNode
    for c in n:
      result.add c.replaceExcl(x,y,k)

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
    result = b

proc append(x,y:NimNode) =
  for c in y: x.add c

proc matchGeneric(n,ty,g:NimNode):NimNode =
  ## Match generic type `ty`, with the instantiation node `n`, and return
  ## the instantiation type of the generic identifier `g`.
  # echo "MG:I: ",n.lisprepr
  # echo "MG:T: ",ty.lisprepr
  # echo "MG:G: ",g.lisprepr
  proc isG(n:NimNode):bool =
    n.kind == nnkIdent and n.eqIdent($g)
  proc typeof(n:NimNode):NimNode =
    newNimNode(nnkTypeOfExpr,g).add n
  proc getGParams(ti:NimNode):NimNode =
    # ti ~ type[G0,G1,...], is from gettypeinst
    # We go through the implementation to find the correct generic names.
    ti.expectKind nnkBracketExpr
    let tn = ti[0]
    tn.expectKind nnkSym
    let td = tn.symbol.getImpl
    td.expectKind nnkTypeDef
    result = td[1]
    result.expectKind nnkGenericParams
  proc matchT(ti,ty,g:NimNode):NimNode =
    # match instantiation type `ti`, with generic type `ty`
    # recursively find the chain of generic type variables
    # correponding to `g` in `ty`.
    result = newPar()
    var i = 0
    let tg = getGParams ti
    while i<ty.len:
      if ty[i].isG: break
      inc i
    if i == 0: return
    elif i < ty.len:
      if tg.len < i: return
      else: result.add tg[i-1]
    else:
      for i in 1..<ty.len:
        if ty[i].kind == nnkBracketExpr:
          if i < ti.len:
            ti[i].expectKind nnkBracketExpr
            result = matchT(ti[i],ty[i],g)
            if result.len > 0:
              result.add tg[i-1]
              return
  if ty.isG: return typeof n
  elif ty.kind == nnkBracketExpr:
    let ts = matchT(n.gettypeinst,ty,g)
    result = n
    if ts.len > 0:
      for i in countdown(ts.len-1,0): result = result.newDotExpr ts[i]
      return
  echo "Internal WARNING: matchGeneric: Unsupported"
  echo "MG:I: ",n.lisprepr
  echo "MG:T: ",ty.lisprepr
  echo "MG:G: ",g.lisprepr

proc cleanIterator(n:NimNode):NimNode =
  var fa = newPar()
  proc replaceFastAsgn(n:NimNode):NimNode =
    if n.kind == nnkFastAsgn:
      let n0 = genSym(nskLet, $n[0].symbol)
      fa.add newPar(n[0],n[1],n0)
      template asgn(x,y:untyped):untyped =
        let x = y
      let n1 = replaceFastAsgn n[1]
      result = getAst(asgn(n0,n1))[0]
    else:
      result = n.copyNimNode
      for c in n: result.add replaceFastAsgn c
  proc removeDeclare(n:NimNode):NimNode =
    if n.kind == nnkVarSection:
      var keep = newseq[int](0)
      for c in 0..<n.len:
        var i = 0
        while i < fa.len:
          var j = 0
          while j < n[c].len-2:
            if fa[i][0] == n[c][j]: break
            inc j
          if j < n[c].len-2:
            if n[c].len > 3:
              echo "Internal ERROR: cleanIterator: removeDeclare: unhandled situation"
              echo n.treerepr
              quit 1
            break
          inc i
        if i < fa.len and (n[c][^2].kind != nnkEmpty or n[c][^1].kind != nnkEmpty):
          echo "Internal ERROR: cleanIterator: removeDeclare: unhandled situation"
          echo n.treerepr
          quit 1
        elif i >= fa.len: keep.add c
      # echo keep," ",n.repr
      if keep.len == 0:
        # echo "Removing declaration: ",n.lisprepr
        result = newNimNode(nnkDiscardStmt,n).add newStrLitNode(n.lisprepr)
      else:
        result = n.copyNimNode
        for i in keep: result.add removeDeclare n[i]
    else:
      result = n.copyNimNode
      for c in n: result.add removeDeclare c
  result = replaceFastAsgn n
  if fa.len > 0:
    result = result.removeDeclare
    for x in fa:
      result = result.replace(x[0],x[2])
      # echo x[0].lisprepr,"\n  :: ",x[0].gettypeinst.lisprepr
      # echo x[1].lisprepr,"\n  :: ",x[1].gettypeinst.lisprepr
  proc fixDeclare(n:NimNode):NimNode =
    # Inlined iterators have var sections that are not clearly typed.
    # We try to find inconsistencies from the type of the actual symbol being declared.
    result = n.copyNimNode
    if n.kind == nnkVarSection:
      for i in 0..<n.len:
        result.add n[i].copyNimTree
        if n[i][^2].kind == nnkEmpty and n[i][^1].kind != nnkEmpty:
          for j in 0..<n[i].len-2:
            # echo n.treerepr
            # echo "sym ",i," ",j," : ",n[i][j].repr
            # echo "    :- ",n[i][^1].repr
            let
              t = n[i][j].gettypeinst
              r = n[i][^1].gettypeinst
            # echo "    ty: ",t.lisprepr
            # echo "    <-: ",r.lisprepr
            # echo "    ??: ",t==r
            if result[i][^2].kind != nnkEmpty and result[i][^2] != r:
              echo "Internal ERROR: cleanIterator: fixDeclare: unhandled situation"
              echo n.treerepr
              quit 1
            # echo "Fixing declaration: ",n[i].lisprepr
            if t != r: result[i][^2] = newNimNode(nnkTypeOfExpr,n[i][j]).add n[i][j]
        result[i][^1] = fixDeclare result[i][^1]
      # echo result.repr
    else:
      for c in n: result.add fixDeclare c
  result = fixDeclare result
  # echo "<<<<<< cleanIterator"
  # echo result.treerepr

proc inlineLets(n:NimNode):NimNode =
  proc get(n:NimNode):NimNode =
    result = newPar()
    if n.kind == nnkLetSection:
      for d in n:
        if d.kind != nnkIdentDefs or d.len<3:
          echo "Internal ERROR: regenSym: get: can't handle:"
          echo n.treerepr
          quit 1
        if d[^1].kind in AtomicNodes:
          for i in 0..<d.len-2:   # Last 2 is type and value.
            if d[i].kind == nnkSym:
              result.add newPar(d[i],d[^1])
            else: error("inlineLets can't handel: " & n.treerepr)
          for c in d[^1]: result.append get c
    else:
      for c in n: result.append get c
  proc rep(n,x,y:NimNode):NimNode =
    if n == x: result = y.copy
    elif n.kind == nnkLetSection:
      var ll = n.copyNimNode
      for d in n:
        var dd = d.copyNimNode
        for i in 0..<d.len-2:
          if d[i] != x: dd.add d[i].copy
        if dd.len > 0:
          dd.add(d[^2].copy, d[^1].rep(x,y))
          ll.add dd
      if ll.len > 0: result = ll
      else: result = newNimNode(nnkDiscardStmt,n).add(newStrLitNode(n.repr))
    else:
      result = n.copyNimNode
      for c in n:
        result.add c.rep(x,y)
  result = n.copy
  for x in get n: result = result.rep(x[0],x[1])

proc regenSym(n:NimNode):NimNode =
  # Only regen nskVar and nskLet symbols.

  # We need to regenerate symbols for multiple inlined procs,
  # because cpp backend put variables on top level, although
  # the c backend works without this.
  proc get(n:NimNode,k:NimNodeKind):NimNode =
    result = newPar()
    if n.kind == k:
      for d in n:
        if d.kind != nnkIdentDefs or d.len<3:
          echo "Internal ERROR: regenSym: get: can't handle:"
          echo n.treerepr
          quit 1
        for i in 0..<d.len-2:   # Last 2 is type and value.
          if d[i].kind == nnkSym: result.add d[i]
        for c in d[^1]: result.append c.get k
    else:
      for c in n: result.append c.get k
  result = n.copyNimTree
  # We ignore anything inside a typeOfExpr, because we need the
  # type information in there, but our new symbols wouldn't have
  # any type info.
  for x in result.get nnkLetSection:
    #echo "Regen Let: ",x.repr
    let y = genSym(nskLet, $x.symbol)
    result = result.replaceExcl(x,y,nnkTypeOfExpr)
  for x in result.get nnkVarSection:
    #echo "Regen Var: ",x.repr
    let y = genSym(nskVar, $x.symbol)
    result = result.replaceExcl(x,y,nnkTypeOfExpr)

proc inlineProcsY*(call: NimNode, procImpl: NimNode): NimNode =
  # echo ">>>>>> inlineProcsY"
  # echo "call:\n", call.lisprepr
  # echo "procImpl:\n", procImpl.treerepr
  let fp = procImpl[3]  # formal params
  proc removeRoutines(n:NimNode):NimNode =
    # We are inlining, so we don't need RoutineNodes anymore.
    if n.kind in RoutineNodes:
      result = newNimNode(nnkDiscardStmt,n).add(newStrLitNode(n.repr))
    else:
      result = n.copyNimNode
      for c in n: result.add removeRoutines c
  proc removeTypeSections(n:NimNode):NimNode =
    # Type section is special.  Once the type is instantiated, it exists, and we don't want duplicates.
    if n.kind == nnkTypeSection:
      result = newNimNode(nnkDiscardStmt,n).add(newStrLitNode(n.repr))
    else:
      result = n.copyNimNode
      for c in n: result.add removeTypeSections c
  var
    pre = newStmtList()
    body = procImpl.body.copyNimTree.removeRoutines.removeTypeSections
  # echo "### body w/o routines:"
  # echo body.repr
  body = cleanIterator body
  # echo "### body after clean up iterator:"
  # echo body.repr
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
    if typ.kind == nnkStaticTy:
      # echo typ.lisprepr
      # echo p.lisprepr
      if p.kind notin nnkLiterals:
        echo "ERROR: inlineProcsY: param type: ",typ.lisprepr
        echo "    received a non-literal node: ",p.lisprepr
        quit 1
      # We do nothing, assuming the compiler has finished constant unfolding.
    elif typ.kind == nnkVarTy:
      pre.add getAst(letX(t, newNimNode(nnkAddr,p).add p))[0]
      body = body.replaceNonDeclSym(sym, newNimNode(nnkDerefExpr,p).add(t), nnkHiddenDeref)
    else:
      pre.add getAst(letX(t, p))[0]
      body = body.replaceNonDeclSym(sym, t)
  # echo "### body with fp replaced:"
  # echo body.repr
  proc resolveGeneric(n:NimNode):NimNode =
    proc find(n:NimNode, s:string):bool =
      if n.kind == nnkDotExpr:
        # ignore n[1]
        return n[0].find s
      elif n.kind in RoutineNodes:
        return false
      elif n.kind == nnkIdent and n.eqIdent s:
        return true
      else:
        for c in n:
          if c.find s: return true
      return false
    var gs = newPar()
    if procImpl[5].kind == nnkBracket and procImpl[5].len>=2 and procImpl[5][1].kind == nnkGenericParams:
      let gp = procImpl[5][1]
      for c in gp:
        c.expectKind nnkIdentDefs
        for i in 0..<c.len-2:
          c[i].expectKind nnkIdent
          if n.find($c[i]): gs.add c[i]
    result = n
    # echo gs.lisprepr
    for g in gs:
      var j = 1
      var sym,typ:NimNode
      while j < call.len:
        (sym,typ) = fp.getParam j
        if typ.find($g): break
        inc j
      if j < call.len:
        # echo sym.treerepr
        # echo typ.treerepr
        # let tyi = call[j].gettypeinst
        # echo "timpl: ",call[j].gettypeimpl.lisprepr
        # echo "tinst: ",tyi.lisprepr
        # echo "impl: ",tyi[0].symbol.getimpl.lisprepr
        let inst = matchGeneric(call[j], typ, g)
        # echo inst.treerepr
        result = result.replaceId(g, inst)
      else:
        echo "Internal WARNING: resolveGeneric: couldn't find ",g.lisprepr
  body = resolveGeneric body
  # echo "### body after resolve generics:"
  # echo body.repr
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
  # echo "### body after replace return with break:"
  # echo body.repr
  var sl:NimNode
  if procImpl.len == 7:
    pre.add body
    sl = newBlockStmt(blockname, pre)
  elif procImpl.len == 8:
    # echo "TYPEof call: ",call.lisprepr," ",call.gettypeinst.treerepr
    # echo "TYPEof call: ",call.lisprepr," ",call.gettypeimpl.treerepr
    # echo "TYPEof call[0]: ",call[0].lisprepr," ",call[0].gettypeinst.treerepr
    # echo "TYPEof call[0]: ",call[0].lisprepr," ",call[0].gettypeimpl.treerepr
    # echo "TYPEof fp[0]: ",fp[0].lisprepr," ",fp[0].gettype.treerepr
    # echo "TYPEof pi[7]: ",procImpl[7].lisprepr," ",procImpl[7].gettype.treerepr
    template varX(x,t:untyped):untyped =
      var x: t
    template varXNI(x,t:untyped):untyped =
      var x {.noinit.} : t
    let
      #ty = call.gettypeinst
      #ty = call[0].gettypeinst[0][0]
      #ty = call[0].gettypeimpl[0][0]
      ty = newNimNode(nnkTypeOfExpr,call).add(call.copyNimTree)
      r = procImpl[7]
      z = genSym(nskVar, $r.symbol)
      p = procImpl[4]
    var noinit = false
    if p.kind != nnkEmpty:
      # echo "pragmas: ", p.lisprepr
      p.expectKind nnkPragma
      for c in p:
        if c.eqIdent "noinit":
          noinit = true
          break
    let d = if noinit: getAst(varXNI(z,ty)) else: getAst(varX(z,ty))
    # if noinit: echo "noinit: ", d.lisprepr
    pre.add body.replace(r,z)
    sl = newBlockStmt(newNimNode(nnkStmtListExpr,call).add(d[0], newBlockStmt(blockname, pre), z))
  else:
    echo "Internal ERROR: inlineProcsY: unforeseen length of the proc implementation: ", procImpl.len
    quit 1
  # echo "====== sl"
  # echo sl.repr
  # echo "^^^^^^"
  # result = sl
  result = regenSym inlineLets sl
  # echo "<<<<<< inlineProcsY"
  # echo result.treerepr

proc callName(x: NimNode): NimNode =
  if x.kind in CallNodes: result = x[0]
  else: quit "callName: unknown kind (" & treeRepr(x) & ")\n" & repr(x)

proc inlineProcsX*(body: NimNode): NimNode =
  # echo ">>>>>> inlineProcsX"
  # echo body.repr
  proc recurse(it: NimNode): NimNode =
    if it.kind == nnkTypeOfExpr: return it.copyNimTree
    if it.kind in CallNodes and it.callName.kind==nnkSym:
      let procImpl = it.callName.symbol.getImpl
      # echo "inspecting call"
      # echo it.lisprepr
      # echo procImpl.repr
      if procImpl.body.kind!=nnkEmpty and
          not isMagic(procImpl) and
          procImpl.kind != nnkIteratorDef:
        return recurse inlineProcsY(it, procImpl)
    result = copyNimNode(it)
    for c in it: result.add recurse c
  result = recurse(body)
  # echo "<<<<<< inlineProcsX"
  # echo result.repr

macro inlineProcs*(body: typed): auto =
  # echo ">>>>>> inlineProcs:"
  # echo body.repr
  # echo body.treerepr
  #result = body
  result = rebuild inlineProcsX body
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

  a1(1.0)
  a2(1.0)
