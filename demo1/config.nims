switch("cc", "gcc")
switch("gcc.cpp.exe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.linkerexe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.options.always", "--x cu -ccbin=g++-4.9")
switch("gcc.cpp.options.speed", "-O3 -Xcompiler -march=haswell,-fPIC")

#switch("gcc.cpp.options.speed", "-O3 -march=haswell")
