--define:release
--threads:on
--tlsEmulation:off
--cc:gcc
switch("gcc.cpp.exe", "./ccwrapper")
switch("gcc.cpp.linkerexe", "./ccwrapper")
switch("gcc.cpp.options.always", "-x cu -std=c++11")
switch("gcc.cpp.options.speed", "-O3 -Xcompiler -Ofast,-march=native")
#switch("gcc.cpp.options.speed", "-O3 -Xcompiler -Ofast,-march=native,-fno-strict-aliasing")
