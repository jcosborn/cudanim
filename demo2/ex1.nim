import cpugpuarray

let N = 64
var x = newColorMatrixArray(N)
var y = newColorMatrixArray(N)
var z = newColorMatrixArray(N)

# set them to diagonal matrices on CPU
x := 1
y := 2
z := 3

# do something on CPU
x += y * z

# do something on GPU
onGpu:
  x += y * z
  z := 4

# do something on CPU again
x += y * z

if x[0][0,0].re == 21.0:
  echo "yay, it worked!"
  echo "do you agree, GPU?"
else:
  echo x[0][0,0].re

onGpu:
  if getThreadNum()==0:
    if x[0][0,0].re == 21.0:
      printf("yes, I agree!\n")

# outputs:
#   yay, it worked!
#   do you agree, GPU?
#   yes, I agree!
