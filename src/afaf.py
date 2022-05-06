"""Approximating functions as functions.

Which appears to be approximating all functions that
are not polynomials as polynomials of some form. This 
is useful and basically required because computers can
*generally* only handle operations associated with 
polynomials, namely, addition, subtraction, multiplication,
and division.

Unlike other modules in the naf package this one is a little
more disjointed. It really needs a robust generic polynomial
class and a Chebyshev polynomial class but I want to remain
focused on the breadth of the general topic of numerical
analysis making notes about depth topics to which to return. 
"""


from sympy import symbols, diff, Poly
from naf.linalg import doqsv
import numpy as np
import math
from numpy.polynomial import Polynomial as P

def chebyshev_symbolic(n):
    """Compute symbolic Chebyshev polynomial of degree n.

    Parameters
    ----------
    n : integer
        Degree of Chebyshev polynomial

    Returns
    -------
    t : sympy expression
        Symbolic Chebyshev polynomial
    """
    x = symbols('x')
    
    t = []
    
    t.append(1)
    t.append(x)
    
    for i in range(1,n+2):
        t.append(simplify(2*x*t[i] - t[i-1]))
        
    return t[n]

def maclaurin(equ, n):
    """Determine the symbolic Maclaurin series of degree n for the given equation.
    
    Uses the sympy library to do symbolic differentation. 
    
    Parameters
    ----------
    equ : sympy expression
        Symbolic representation of the equation to be approximated by the Maclaurin
        series. The expression must be written in terms of the variable 'x'.
    n : integer
        Degree of Maclaurin polynomial
        
    Returns
    -------
    mac : sympy expression
        Symbolic representation of the approximate equivalent Maclaurin series
        
    """
    x,z = symbols('x z')
    
    mac = 0
    for i in range(0,n+1):
        mac += (diff(equ,x,i)*z**i)/math.factorial(i)
    return mac

def pade(n,m,c):
    """Calculates the Pade approximation coefficients
    
    
    Paramters
    ---------
    n : integer
        Degree of polynomial in numerator
    m : integer
        Degree of polynomial in denominator
    c : 1D numpy array
        Maclaurin series coefficients up to degree N = n+m
        in ascending order of degree, matching numpy polynomial ordering

        c3*x^3 + c2*x^2 + c1*x + c0
        [c0, c1, c2, c3, ..., cn]
    
    Returns
    -------
    a : numpy Polynomial
        Pade numerator coefficients in ascending order
        [a0,a1,a2,a3,...,an]
    b : numpy Polynomial
        Pade denominator coefficients in ascending order
        [1,b1,b2,b3,...,bm]

    Notes:
    1. This function only works for n==m or n == m+1 which is a
    typical application of the Pade approximation       
    """
    cb = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            cb[i,j] = c[n-m+(j+i+1)]

    d = np.empty(m)
    for i in range(m):
        d[i] = -1*c[n+i+1]

    b = doqsv(cb, d)

    b = np.flip(np.append(b,1))

    a = np.empty(n+1)
    ca = np.zeros((n+1,m+1))

    for i in range(n+1):
        for j in range(m+1):
            if j<=i:
                ca[i,j] = c[i-j]

    a = np.matmul(ca,b)
    
    return a, b

def get_npp_coeffs(equ, deg):
    """Gets numpy polynomial coefficients.
    
    The sympy polynomial class function all_coeffs() truncates leading zero-value
    coefficients which are necessary information for certain algorithms which use
    the coefficents up to a specific degree polynomial.
    
    Initially sets all coefficients to zero then replaces the coefficient with the
    polynomial coefficient value if it exists.
    
    Parameters
    ----------
    equ : sympy expression
        Sympy polynomial equation
    deg : integer
        Required degree of polynomial; required number of coefficients
        
    Returns
    -------
    c : 1D numpy array
        Array of the correct number of coefficients for the requested degree.
        Coefficients are arranged in ascending order of degree, this matches
        numpy ordering

        c3*x^3 + c2*x^2 + c1*x + c0
        [c0, c1, c2, c3]
        
    Example
    -------
    """
    c = np.zeros(deg+1)
    i = 0
    for cf in np.flip(Poly(equ).all_coeffs()):
        c[i] = cf
        i += 1
        
    return c

###########################
#Continued Fractions
#########################


def poly_min_degree(p):
    """Returns the minimum degree of the first non-zero coefficient.
    
    Parameters
    ----------
    p : numpy Polynomial
        polynomial to evaluate
        
    Returns
    -------
    i : integer
        minimum degree of first non-zero coefficient
    """
    
    for i in range(len(p)+1):
        if p.coef[i] != 0:
            break
    
    return i

def reduce_poly_degree(p, md):
    """Returns a polynomial of reduced degree.
    
    Function reduces the polynomial degree by the value
    of md. 
    
    Parameters
    ----------
    p : numpy Polynomial
        polynomial to be reduced
    md : integer
        degree of reduction
        
    Returns
    -------
    p : numpy Polynomial
        reduced degree polynomial
        
    Example
    -------
    function = x^4 + 5x^3 + 7x^2
    
    p = P([0,0,7,5,4])
    md = 2
    p = P([7,5,4])
    
    function = x^2 + 5x + 7
    """
    
    if md != 0:
        rd = p.degree() - md

        for i in range(md, len(p)):
            p.coef[i-md] = p.coef[i]

        p = p.cutdeg(rd)
        
    return p

def continued_fraction(pn, pd):
    """Calculates the continued fraction from a rational function.
    
    Parameters
    ----------
    pn : numpy Polynomial
        numerator polynomial
    pd : numpy Polynomial
        denomiator polynomial
        
    Returns
    -------
    factors : array of numpy Polynomials
        polynomials factored out of numerator
    quos : array of numpy Polynomials
        quotients computed from dividing rational polynomials
    rem : numpy Polynomial
        the last remainder value; this should be a 0th degree polynomial
    p2 : numpy Polynomial
        the last denominator polynomial when the while-loop exits
    """
    
    factors = []
    quos = []

    rd = 1 #starting value
    p1 = pn
    p2 = pd

    while rd > 0:
        normalizer = p2.coef[-1]
        p1 = p1/normalizer
        p2 = p2/normalizer
        md = poly_min_degree(p1)
        p1 = reduce_poly_degree(p1,md)
        factor = p1.coef[-1]
        p1 /= factor
        pa = [0.0 for i in range(md)]
        pa.append(factor)
        factors.append(P(pa))

        quo, rem = divmod(p2, p1)

        quos.append(quo)

        p2 = p1
        p1 = rem

        rd = rem.degree()

    return factors, quos, rem, p2

def str_repr_continued_fraction(factors, quos, rem, ld):

    p_out = str(rem) + "/" + "(" + str(ld) + ")"

    for i in range(len(factors)-1,-1,-1):
        p_out = "(" + str(factors[i]) + ")" + "/ (" + "(" + str(quos[i]) + ") +" + p_out + ")"

    return p_out

def expanded_synthetic_division(dividend, divisor):
    """Polynomial division using Expanding Synthetic division
    
    This method works for both monic and non-monic rational polynomials.
    
    a_n*x^n + a_n-1*x^n-1 + a_n-2*x^n-2 + ... + a_0*x^0
    [a_n, a_n-1, a_n-2, ... , a_0]
    
    Paramters
    ---------
    dividend : 1D numpy array
        Polynomial coefficients of the dividend (numerator) in order
        of decreasing power 
    divisor : 1D numpy array
        Polynomial coefficients of the divisor (denominator) in order
        of decreasing power
        
    Returns
    -------
    quotient : 1D numpy array
        Polynomial coefficients of the quotient of the polynomial
        division in order of decreasing power
    remainder : 1D numpy array
        Polynomial coefficients of the remainder of the polynomial
        division in order of decreasing power
        
    """
    
    #copy dividend so original values are preserved
    out = np.copy(dividend) 
    
    normalizer = divisor[0]
    
    
    for i in range(len(dividend) - len(divisor) + 1):
        out[i] = out[i] / normalizer
        
        coeff = out[i]
        if coeff != 0:
            for j in range(1, len(divisor)):
                out[i+j] += -divisor[j]*coeff
                
    separator = 1 - len(divisor)
    return out[:separator],out[separator:]
