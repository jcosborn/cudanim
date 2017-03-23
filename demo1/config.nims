switch("cc", "gcc")
switch("gcc.cpp.exe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.linkerexe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.options.always", "--x cu")
switch("gcc.cpp.options.speed", "-O3 -Xcompiler -march=haswell")

#switch("gcc.cpp.options.speed", "-O3 -march=haswell")
