--define:release
--threads:on
--tlsEmulation:off
--cc:gcc
const USEGPU {.intdefine.} = 1
when USEGPU == 0:
  switch("gcc.exe", "gcc")
  switch("gcc.linkerexe", "gcc")
  switch("gcc.options.always", "-std=c11")
  switch("gcc.options.speed", "-Ofast -march=native")
else:
  switch("gcc.cpp.exe", "./ccwrapper")
  switch("gcc.cpp.linkerexe", "./ccwrapper")
  switch("gcc.cpp.options.always", "-x cu -std=c++11")
  switch("gcc.cpp.options.speed", "-O3 -Xcompiler -Ofast,-march=native")
  #switch("gcc.cpp.options.speed", "-O3 -Xcompiler -Ofast,-march=native,-fno-strict-aliasing")
