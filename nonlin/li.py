import math
import numpy

def lin_intp(f, x0, x1, tol = 0.001, iter_limit = 20):
    """
    Root finding method using linear interpolation (false position).

    The linear interpolation method ensures the existance of root by approaching
    it from both sides, i.e. above and below the x-axis. The algorithem keeps
    the root bracketed for each successive iteration. 

    Args:
        f (function): mathematical function to evalue in terms of a single
        variable
        x0 (float)*: first guess
        x1 (float)*: second guess
        tol (Optional[float]): solution toleration to analytically correct
        answer. Defaults to 0.001
        iter_limit (Optional[int]): maximum number of iterations to persom if
        the solution is not converging. Defaults to 20. 

        *Note: The values x0 and x1 must bracket the root. 

    Returns:
        [float, float] if successful.

        The first index is the approximate root of the equation determined by
        the linear interpolation method and the second index is the value of the
        function at the approximated root. 

    Raises:

    """

    iter = 1 #initialize iteration counter

    x2 = x1 #dummy value

    while f(x2) > tol or iter > iter_limit:
        f0 = f(x0)
        f1 = f(x1)

        x2 = x1 - f1*(x0-x1)/(f0-f1)
        f2 = f(x2)

        #if f2 is on the opposite side of the x-axis as f0
        #x1 is updated to equal x2 since x2 is close to the root;
        #else f2 is on the same side of the x-axis as f0
        #x0 is updated to equal x2
        if numpy.sign(f2) != numpy.sign(f0):
            x1 = x2
        else:
            x0 = x2

    return x2, f2

if __name__ == "__main__":
    f = lambda x: 3*x + math.sin(x) - math.exp(x)

    x0 = 0
    x1 = 1

    print lin_intp(f, x0, x1)
