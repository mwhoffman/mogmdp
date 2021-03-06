## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>
## Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy
## Further modifications by M. Hoffman.

## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.


import numpy as np
import scipy.optimize._lbfgsb as _lbfgsb

def lbfgsb(func, x0,
           bounds=None, m=10, factr=1e7, pgtol=1e-5, maxfun=100):
    """
    Minimize a function func using the L-BFGS-B algorithm.

    Arguments:

    func    -- function to minimize. Called as func(x, *args)
    x0      -- initial guess to minimum
    bounds  -- a list of (min, max) pairs for each element in x, defining
               the bounds on that parameter. Use None for one of min or max
               when there is no bound in that direction
    m       -- the maximum number of variable metric corrections
               used to define the limited memory matrix. (the limited memory BFGS
               method does not store the full hessian but uses this many terms in an
               approximation to it).
    factr   -- The iteration stops when
                   (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
               where epsmch is the machine precision, which is automatically
               generated by the code. Typical values for factr: 1e12 for
               low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
               high accuracy.
    pgtol   -- The iteration will stop when
                   max{|proj g_i | i = 1, ..., n} <= pgtol
               where pg_i is the ith component of the projected gradient.
    maxfun  -- maximum number of function evaluations.

    License of L-BFGS-B (Fortran code)
    ==================================

    The version included here (in fortran code) is 2.1 (released in 1997). It was
    written by Ciyou Zhu, Richard Byrd, and Jorge Nocedal <nocedal@ece.nwu.edu>. It
    carries the following condition for use:

    This software is freely available, but we expect that all publications
    describing  work using this software , or all commercial products using it,
    quote at least one of the references given below.

    References
     * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
       Constrained Optimization, (1995), SIAM Journal on Scientific and
       Statistical Computing , 16, 5, pp. 1190-1208.
     * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
       FORTRAN routines for large scale bound constrained optimization (1997),
       ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550 - 560.
    """
    n = len(x0)

    if bounds is None: bounds = [(None,None)] * n
    if len(bounds) != n: raise ValueError('length of x0 != length of bounds')

    nbd = np.zeros((n,), np.int32)
    low_bnd = np.zeros((n,), np.float64)
    upper_bnd = np.zeros((n,), np.float64)
    bounds_map = {(None, None): 0, (1, None): 1, (1, 1): 2, (None, 1): 3}

    for i in range(0, n):
        l,u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    wa  = np.zeros((2*m*n + 4*n + 12*m**2 + 12*m,), np.float64)
    iwa = np.zeros((3*n,), np.int32)
    task  = np.zeros(1, 'S60')
    csave = np.zeros(1, 'S60')
    lsave = np.zeros((4, ), np.int32)
    isave = np.zeros((44,), np.int32)
    dsave = np.zeros((29,), np.float64)

    # allocate space for our path.
    ns = np.empty(maxfun+1, np.int32)
    xs = np.empty((maxfun+1, n), np.float64)
    fs = np.empty(maxfun+1, np.float64)
    gs = np.empty((maxfun+1, n), np.float64)

    # initialize the first step.
    x = np.array(x0, np.float64)
    f, g = func(x)
    ns[0], xs[0], fs[0], gs[0] = 0, x, f, g
    i = 1
    numevals = 0
    task[:] = 'START'

    while 1:
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, -1, csave, lsave,
                       isave, dsave)
        task_str = task.tostring()
        if task_str.startswith('FG'):
            # minimization routine wants f and g at the current x
            numevals += 1
            f, g = func(x)

        elif task_str.startswith('NEW_X'):
            # new iteration
            ns[i], xs[i], fs[i], gs[i] = numevals, x, f, g
            i += 1
            if numevals > maxfun:
                task[:] = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        else:
            break

    ns.resize(i)
    xs.resize((i, n))
    fs.resize(i)
    gs.resize((i, n))

    return x, f, dict(numevals=ns, x=xs, f=fs, g=gs)

