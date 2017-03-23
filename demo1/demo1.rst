===========================
Portable expressions in Nim
===========================

:Author: James C. Osborn

.. contents::

Preliminaries
=============

This document was created with Nim's built-in documentation generator.
It can parse documentation comments in the source code and also process
separate reStructuredText_ files.
This document was made from a reStructuredText file using Nim's
document generator to try it out and also take advantage of its Nim
code highlighter.

.. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText

Code portability in Nim
=======================

Here's an example of the result (so far)

.. code-block:: Nim

  import cpugpuarray

  let N = 1000
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

  onGpu:
    if getThreadNum()==0:
      if x[0][0,0].re == 21.0:
        printf("yes, I agree!\n")

  # outputs:
  #   yay, it worked!
  #   do you agree, GPU?
  #   yes, I agree!


The above can be compiled and run with

::

  nim cpp -d:release -r ex1.nim


This is basically the main interface that the average user would need to
deal with, the rest is just details for the curious.


Implementation details
======================

The main container object in the example above is an array that can live
on the CPU and also the GPU.  This is defined as

.. code-block:: Nim

  type
    ArrayObj*[T] = object
      p*: ptr array[0,T]
      n*: int
      g*: GpuArrayObj[T]
      lastOnGpu*: bool

    GpuArrayObj*[T] = object
      p*: ptr array[0,T]
      n*: int

``ArrayObj[T]`` is a generic array-like object parameterized on the type ``T``.
This is similar to a templated type declaration in C++ with ``T`` being the template parameter (Nim uses ``[T]`` instead of ``<T>`` for generics).
The ``*`` (star) after all the type and field names above means that they are exported from this module (file).
They will be visible to another module that ``import``'s this module (otherwise they would be private to this module).

The ``ArrayObj`` contains four fields:

- ``p``: which is a pointer (``ptr``) for the data on the host. \
This is implemented as a pointer to an array of length ``0`` \
with elements of type ``T`` for convenience. \
This should really be marked with an ``{.unchecked.}`` pragma to prevent \
bounds checking in debug mode (bounds checks are off by default in release mode).
- ``n``: the number of elements in the array.
- ``g``: a GPU array object, defined next.
- ``lastOnGpu``: a Boolean that tells us which pointer is valid.

The ``GpuArrayObj`` is similar to ``ArrayObj``, but just contains a pointer \
(which will hold a GPU pointer) and the number of elements.
This is the object we will pass to the GPU, so it contains a copy of the \
length for convenience.


Offloading
==========

The offload magic happens in the ``onGpu:`` block.
It is defined like

.. code-block:: Nim

  # the default total threads (nn=32*256) and threads per block (tpb=256)
  # are just for testing, they really should be an educated
  # guess made from querying the device
  template onGpu*(body: untyped): untyped = onGpu(32*256, 256, body)

This launches a CUDA kernel using the default number of threads and threads \
per block.  Right now they are hard-coded, but should really come from \
querying the device (or let the user specify some global default).

One can override the defaults for a call by explicitly specifying them

.. code-block:: Nim

  onGpu(x.n, 128):
    x += y * z
    z := 4

This would launch one (virtual) thread per element of the array ``x`` and use
128 threads per block.

The CUDA kernel gets created here

.. code-block:: Nim

  template onGpu*(nn,tpb: untyped, body: untyped): untyped =
    block:
      var v = packVars(body, getGpuPtr)
      type myt {.bycopy.} = object
	d: type(v)
      proc kern(xx: myt) {.cudaGlobal.} =
	template deref(k: int): untyped = xx.d[k]
	substVars(body, deref)
      let ni = nn.int32
      let threadsPerBlock = tpb.int32
      let blocksPerGrid = (ni+threadsPerBlock-1) div threadsPerBlock
      cudaLaunch(kern, blocksPerGrid, threadsPerBlock, v)
      discard cudaDeviceSynchronize()

This starts a new block scope (``block:``), similar to ``{...}`` in C.
This is done to isolate the defined kernel (``proc kern ...``) from other \
``onGpu`` blocks.

The first major task is to examine the body of the ``onGpu`` block and \
extract the variables that are used.
This is done by the ``packVars`` macro.
It walks the syntax tree of the code block passed in and keeps track of \
the (unique) variables it references.
It then spits out a data structure (a tuple_) containing those variables.
It wraps each variable in a call to the function name that was passed in \
(in this case ``getGpuPtr``).
For the example above, this line would get expanded to

.. _tuple: https://nim-lang.org/docs/manual.html#types-tuples-and-object-types

.. code-block:: Nim

  var v = (getGpuPtr(x), getGpuPtr(y), getGpuPtr(z))

The function ``getGpuPtr`` can then be defined independently for each type \
to return a valid GPU object (it actually doesn't have to be a pointer as we'llsee next).
For the ``ArrayObj`` type it is defined as

.. code-block:: Nim

  template getGpuPtr*(x: var ArrayObj): untyped =
    toGpu(x)
    x.g

This copies the data to the GPU (if necessary) and then returns the \
``GpuArrayObj`` containing the GPU pointer and the length of the array.
This is a (small) object residing in CPU memory, and the CUDA library \
takes care of copying it to the GPU when passed as an argument.

Copying the data to the GPU is handled by

.. code-block:: Nim

  proc toGpu*(x: var ArrayObj) =
    if not x.lastOnGpu:
      x.lastOnGpu = true
      if x.g.n==0: x.g.init(x.n)
      let err = cudaMemcpy(x.g.p, x.p, x.n*sizeof(x.T), cudaMemcpyHostToDevice)
      if err: echo err

Here we check if this array was last used on the GPU.
If not we check if it has been initialized yet (``x.g.n==0``) and \
initialize it if not (which will call cudaMalloc).
We then copy the CPU memory to GPU memory.
Here we could also translate the layout if we wanted.

Currently I am not distinguishing between read access and write access.
This could lead to further optimization.
It should be possible to modify the existing methods to handle that too.

Next we create the CUDA kernel (``kern``).
The kernel is defined here

.. code-block:: Nim

  proc kern(xx: myt) {.cudaGlobal.} =
    template deref(k: int): untyped = xx.d[k]
    substVars(body, deref)

This is a function taking one argument (which contains the packed \
``GpuArrayObj``'s or any other objects used by the expressions.
I originally wrote the procedure definition as

.. code-block:: Nim

  proc kern(xx: type(v)) {.cudaGlobal.} =
    template deref(k: int): untyped = xx[k]
    substVars(body, deref)

but found that Nim decided in some cases to pass the argument of \
``kern`` (``xx``) as a pointer, instead of by value.
Nim does this to optimize function calls when it feels it is safe to do so.
To prevent this I wrapped the tuple in another object type (``myt``) that \
is explicitly declared ``{.bycopy.}``, so that Nim will always pass it by \
value (which makes a copy).

In retrospect, another approach may have been to mark the procedure as \
``{.exportC.}``, which will also prevent Nim from changing the calling \
conventions.  I would then need to make the procedure names ``kern`` unique \
on my own since Nim will also not perform name-mangling on ``{.exportC.}`` \
procedures.

The main body of the kernel comes from the

.. code-block:: Nim

  substVars(body, deref)

macro.
It works similarly to the ``packVars`` macro above, but this time it will \
identify the variables referenced in the code block and substitute them \
with a call to the provided function (``deref``) with an integer argument \
that specifies which position in the kernel argument tuple that variable \
is in.  For the example above this would generate

.. code-block:: Nim

  deref(0) += deref(1) * deref(2)
  deref(2) := 4

The ``deref`` template then simply expands to the appropriate expression \
that refers to the kernel argument.

The rest of the magic needed to transform this procedure into a valid CUDA \
kernel is handled in the macro ``cudaGlobal`` which is applied to the \
procedure as a pragma ``{.cudaGlobal.}``.
It also performs function inlining, so that one can still call host functions \
from the device (and not have to worry about marking then with ``__device__``.
I won't go into the details here.

The main step left now is to launch the kernel

.. code-block:: Nim

  let ni = nn.int32
  let threadsPerBlock = tpb.int32
  let blocksPerGrid = (ni+threadsPerBlock-1) div threadsPerBlock
  cudaLaunch(kern, blocksPerGrid, threadsPerBlock, v)

This selects the blocksPerGrid and threadsPerBlock to be used in the CUDA \
kernel, then launches the kernel ``kern`` with the argument tuple ``v``.

Lastly, we synchronize.

.. code-block:: Nim

  discard cudaDeviceSynchronize()

This returns an error code, which I really should be checking instead \
of discarding.
Nim requires you to explicitly discard a return value to be clear that you \
meant to ignore it and didn't just forget.
We may be able to delay this until we actually use the fields again.


Back and forth
==============

To get the expression to evaluate correctly on the CPU again we \
also check on every assignment made on the CPU that the fields are \
updated there.  So in the expression

.. code-block:: Nim

  # do something on CPU again
  x += y * z

the ``+=`` will do something like ``packVars``, but this time will generate \
statements containing ``toCpu`` calls on the used variables.

To do
=====

This is just a toy example.

The next step is to get the vectorization working properly on the GPU \
arrays.
The explicit copy allows us to use a different vectorization layout between \
the CPU and GPU.

The examples here also need to be integrated with the existing ``thread:`` \
block in QEX_.
One possibility is simply

.. _QEX: https://github.com/jcosborn/qex

.. code-block:: Nim

  threads:
    # do something on CPU
    x += y * z

    # do something on GPU
    onGpu:
      x += y * z
      z := 4

    # do something on CPU again
    x += y * z

Other variants are also possible.
