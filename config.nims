switch("cc", "gcc")
switch("gcc.cpp.exe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.linkerexe", "/usr/local/cuda/bin/nvcc")
switch("gcc.cpp.options.always", "--x cu -ccbin=gcc-5")
switch("gcc.cpp.options.speed", "-O3")
