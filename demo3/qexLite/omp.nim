import os

when defined(noOpenmp):
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0
  template ompPragma(p:string):untyped = discard
  template setupGc = discard
else:
  const OMPFlag {.strDefine.} = "-fopenmp"
  {. passC: OMPFlag .}
  {. passL: OMPFlag .}
  {. pragma: omp, header:"omp.h" .}
  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}
  template ompPragma(p:string):untyped =
    {. emit:"\n#pragma omp " & p .}
  template setupGc =
    if(omp_get_thread_num()!=0): setupForeignThreadGc()

template ompBarrier* = ompPragma("barrier")
template ompBlock(p:string; body:untyped):untyped =
  ompPragma(p)
  block:
    body

template ompParallel*(body:untyped):untyped =
  ompBlock("parallel"):
    setupGc()
    body
template ompMaster*(body:untyped):untyped = ompBlock("master", body)
template ompSingle*(body:untyped):untyped = ompBlock("single", body)
template ompCritical*(body:untyped):untyped = ompBlock("critical", body)

when isMainModule:
  proc test =
    echo "main: ", ompGetThreadNum(), "/", ompGetNumThreads()
    ompParallel:
      echo "parallel: ", ompGetThreadNum(), "/", ompGetNumThreads()
      ompBarrier()
      ompMaster:
        echo "master: ", ompGetThreadNum(), "/", ompGetNumThreads()
      echo "parallel: ", ompGetThreadNum(), "/", ompGetNumThreads()
      ompSingle:
        echo "single: ", ompGetThreadNum(), "/", ompGetNumThreads()
      echo "parallel: ", ompGetThreadNum(), "/", ompGetNumThreads()
      ompCritical:
        echo "critical: ", ompGetThreadNum(), "/", ompGetNumThreads()
      echo "parallel: ", ompGetThreadNum(), "/", ompGetNumThreads()
      ompBarrier()
  test()
