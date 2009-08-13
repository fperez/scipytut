#!/usr/bin/env python
from __future__ import with_statement
"""Minimal test script to check for modules needed in scipy tutorials.

Execute this code at the command line by typing:

    python adv_tut_checklist.py

If it does NOT say 'OK' at the end, copy the *entire* output of the run and
send it to the course instructor for help.

Note: it NEEDS nose to run at all, so if you don't have nose, it will fail
completely to even start.
"""

# Standard library imports
import glob
import os
import platform
import shutil
import sys
import tempfile

from StringIO import StringIO
from textwrap import dedent

# Third-party imports
import nose
import nose.tools as nt

##############################################################################
# Code begins

#-----------------------------------------------------------------------------
# Generic utility functions
def sys_info():
    """Summarize some info about the system"""

    print '=================='
    print 'System information'
    print '=================='
    print 'os.name      :',os.name
    try:
        print 'os.uname     :',os.uname()
    except:
        # Unix-only
        pass
    print 'platform     :',sys.platform
    print 'platform+    :',platform.platform()
    print 'prefix       :',sys.prefix
    print 'exec_prefix  :',sys.exec_prefix
    print 'executable   :',sys.executable
    print 'version_info :',sys.version_info
    print 'version      :',sys.version
    print '=================='


def c(cmd):
    """Run system command, raise SystemExit if it returns an error."""
    print "$",cmd
    stat = os.system(cmd)
    #stat = 0  # Uncomment this and comment previous to run in debug mode
    if stat:
        raise SystemExit("Command %s failed with code: %s" % (cmd, stat))


def validate_mpl(m):
    """Called if matplotlib imports.  Sets backend for further examples."""
    m.use('Agg')
    # Set this so we can see the big math title in one of the test images
    m.rcParams['figure.subplot.top'] = 0.85


def validate_cython(cython):
    """Called if Cython imports, does further version checks."""
    from Cython.Compiler.Version import version

    min_version = (0, 11, 2)
    cython_version = tuple([int(x) for x in version.split('.')])
    msg = "Cython version %s found, at least %s required" % (cython_version,
                                                             min_version)
    nt.assert_true(cython_version >= min_version, msg)


def validate_sympy(sympy):
    min_version = '0.6.4'
    version = sympy.__version__
    if version < min_version:
        raise ValueError("Sympy version %s, at least %s required" %
                         (version, min_version))
    
#-----------------------------------------------------------------------------
# Tests

def check_import(mnames, validator=None):
    """Check that the given module name can be imported.

    If more than one name is given, all but the last are allowed to raise
    exceptions which are ignored.  Basically, this means that any of the names
    is considered to satisfy the group, and we only report an error if they all
    fail to import.

    A validator is a function that will be called with the imported module
    object if the import succeeds.  It can either do extra tests on the module
    (such as version checks) or simply configure it.
    """

    if isinstance(mnames, basestring):
        exec "import %s as m" % mnames
    else:
        group, last = mnames[:-1], mnames[-1]
        for mname in group:
            try:
                exec "import %s as m" % mname
            except:
                pass
            else:
                break # We exit on first success
        else:
            # If we exhausted all, we try the last one without swallowing errors
            exec "import %s as m" % last

    if validator is not None:
        validator(m)

    # Try to collect some version information
    for vname in ['__version__', 'version', 'Version']:
        try:
            vinfo = eval("m.%s" % vname)
            break
        except AttributeError:
            pass
    else:
        vinfo = '*no info*'
    print 'MOD: %s, version: %s' % (m.__name__,vinfo)


# Test generators are best written without docstrings, because nose can then
# show the parameters being used.
def test_imports():
    modules = ['setuptools',
               'IPython',
               'numpy','scipy','scipy.io',
               'matplotlib','pylab',
               'enthought.mayavi.api',
               # From here on, extra things only in the advanced tutorial
               'scipy.weave',
               'sympy',
               'Cython', 'Cython.Distutils.build_ext',
               'enthought.traits.api',
               'enthought.traits.ui.api',
               ('enthought.traits.ui.wx','enthought.traits.ui.qt4'),
               ]

    validators = dict(matplotlib = validate_mpl,
                      sympy = validate_sympy)

    for mname in modules:
        yield (check_import, mname, validators.get(mname))


def test_weave():
    "Simple code compilation and execution via scipy's weave"
    from scipy import weave

    compiler = 'mingw32' if sys.platform == 'win32' else ''
    
    weave.inline('int x=1;x++;', compiler=compiler)
    
    n,m = 1,2
    code="""
    int m=%s;
    return_val=m+n;
    """ % m
    val = weave.inline(code,['n'], compiler=compiler)
    nt.assert_equal(val,m+n)    


def test_cython():
    """Test basic Cython sanity"""

    # Check the version string
    validate_cython(None)
    
    setup_code = dedent("""
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    import numpy as np

    setup(
        name='Test of proper Cython installation',
        cmdclass={'build_ext': build_ext},
        ext_modules=[ 
            Extension("cython_check",
                      ["cython_check.pyx"], 
                      include_dirs=[np.get_include()])])
                  """)

    cython_code = dedent("""
    cimport cython
    cimport numpy as np
    import numpy as np

    @cython.wraparound(True) # only present in Cython 0.11.2+
    def func():
        cdef np.ndarray[np.int_t] arr = np.arange(2000, 2100, dtype=np.int)
        return "Hello at SciPy %d" % arr[9]
    """)
    
    # Automatically write the code to local temp files
    csetup_fname = 'cython_setup.py'
    cython_fname = 'cython_check.pyx'
    with open(csetup_fname,'w') as f:
        f.write(setup_code)
    with open(cython_fname,'w') as f:
        f.write(cython_code)

    # Try to do the cython build/import/run.
    # XXX - I tried importing the setup function in-process but for some
    # mysterious reason I can't understand, it fails on Python 2.6 (it works on
    # 2.5). So for now, just call it in an external process.
    c('python cython_setup.py build_ext --inplace')
    import cython_check
    result = cython_check.func()
    nt.assert_equal("Hello at SciPy 2009", result)


# Test generator, don't put a docstring in it
def test_loadtxt():
    import numpy as np
    import numpy.testing as npt

    # Examples taken from the loadtxt docstring
    array = np.array
    
    c = StringIO("0 1\n2 3")
    a1 = np.loadtxt(c)
    a2 = np.array([[ 0.,  1.],
                   [ 2.,  3.]])
    yield npt.assert_array_equal,a1,a2

    d = StringIO("M 21 72\nF 35 58")
    a1 = np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
                         'formats': ('S1', 'i4', 'f4')})
    
    a2 = np.array([('M', 21, 72.0), ('F', 35, 58.0)],
                  dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])
    yield npt.assert_array_equal,a1,a2

    c = StringIO("1,0,2\n3,0,4")
    x,y = np.loadtxt(c, delimiter=',', usecols=(0,2), unpack=True)
    yield npt.assert_array_equal,x,np.array([ 1.,  3.])
    yield npt.assert_array_equal,y,np.array([ 2.,  4.])


def test_plot():
    "Simple plot generation."
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot([1,2,3])
    plt.xlabel('some numbers')
    plt.savefig('tmp_test_plot.png')


def test_plot_math():
    "Plots with math"
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot([1,2,3],label='data')
    t=(r'And X is $\sum_{i=0}^\infty \gamma_i + \frac{\alpha^{i^2}}{\gamma}'
       r'+ \cos(2 \theta^2)$')
    plt.title(t)
    plt.legend()
    plt.grid()
    plt.savefig('tmp_test_plot_math.png')


def main():
    """Main routine, executed when this file is run as a script """

    # Create a tempdir, as a subdirectory of the current one, which all tests
    # use for their storage.  By default, it is removed at the end.
    global TESTDIR
    cwd = os.getcwd()
    TESTDIR = tempfile.mkdtemp(prefix='tmp-testdata-',dir=cwd)
    
    print "Running tests:"
    # This call form is ipython-friendly
    try:
        os.chdir(TESTDIR)
        nose.runmodule(argv=[__file__,'-vvs'],
                       exit=False)
    finally:
        os.chdir(cwd)

    print
    print "Cleanup - removing temp directory:", TESTDIR
    # If you need to debug a problem, comment out the next line that cleans
    # up the temp directory, and you can see in there all temporary files
    # created by the test code
    shutil.rmtree(TESTDIR)

    print """
***************************************************************************
                           TESTS FINISHED
***************************************************************************

If the printout above did not finish in 'OK' but instead says 'FAILED', copy
and send the *entire* output, including the system information below, for help.
We'll do our best to assist you.  You can send your message to the Scipy user
mailing list:

    http://mail.scipy.org/mailman/listinfo/scipy-user

but feel free to also CC directly: Fernando.Perez@berkeley.edu
"""
    sys_info()

    
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
