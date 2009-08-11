#!/usr/bin/env python

from __future__ import with_statement

"""Minimal test script to check for modules needed in PyCuda tutorial at Scipy09

Execute this code at the command line by typing:

    python pycuda_tut_checklist.py

If it does NOT say 'OK' at the end, copy the *entire* output of the run and
send it to the course instructor for help.

Note: it NEEDS nose to run at all, so if you don't have nose, it will fail
completely to even start.
"""

# Advanced Tutorial script imports
from adv_tut_checklist import sys_info, c, check_import, main

################################4##############################################
# Code begins

def validate_pycuda(pycuda):
    """Called if PyCuda imports, does further version checks."""
    min_version = 0, 93
    version = pycuda.VERSION
    if version < min_version:
        raise ValueError("PyCuda version %s, at least %s required" %
                         (version, min_version))

# XXX
def validate_cheetah(Cheetah):
    """Called if Cheetah imports, does further version checks."""

    min_version = (2,2,1)
    version = Cheetah.VersionTuple[:3]
    if version < min_version:
        raise ValueError("Cheetah version %s, at least %s required" %
                         (version, min_version))


# Test generators are best written without docstrings, because nose can then
# show the parameters being used.
def test_imports():
    modules = ['setuptools',
               'IPython',
               'numpy','scipy',
               'pycuda', 'pycuda.autoinit',
               'Cheetah',
               ]

    validators = dict(pycuda = validate_pycuda,
                      Cheetah = validate_cheetah,
                      )

    for mname in modules:
        yield (check_import, mname, validators.get(mname))

def test_pycuda():
    """Test basic PyCuda sanity"""

    import pycuda.autoinit
    import pycuda

    import numpy
    import numpy.testing as npt

    mod = pycuda.compiler.SourceModule("""
    __global__ void multiply_them(float *out, float *in1, float *in2)
    {
      const int i = threadIdx.x;
      out[i] = in1[i] * in2[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    arr1 = numpy.random.randn(400).astype(numpy.float32)
    arr2 = numpy.random.randn(400).astype(numpy.float32)

    result = numpy.zeros_like(arr1)
    multiply_them(
        pycuda.driver.Out(result),
        pycuda.driver.In(arr1), pycuda.driver.In(arr2),
        block=(400,1,1))

    npt.assert_array_equal(arr1 * arr2, result)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
