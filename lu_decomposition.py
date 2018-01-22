import numpy as np

n,m = A.shape

#row reduction
for j in xrange(n-1):
    pvt = abs(A[j,j])
    pivot[j] = j
    ipvt_temp = j

    for i in xrange(j+1,n):
        if abs(A[i,j]) > pvt:
            pvt = abs(A[i,j])
            ipvt_temp = i

    if pivot[j] != ipvt_temp:
        switch_rows(j, ipvt_temp)

    for i in xrange(j+1,n):
        A[i,j] = A[i,j]/A[j,j]

    for i in xrange(j+1,n):
        for k in xrange(j+1,n):
            a[i,k] = a[i,k] - a[i,j]*a[j,k]
        b[i] = b[i] - a[i,j]*b[j]

#back substitution
x[n] = b[n]/a[n,n]
for j in xrange(n-1,1,-1):
    x[j] = b[j]
    for k in xrange(n,j+1,-1):
        x[j] = x[j] - x[k]*a[j,k]
    x[j] = x[j]/a[j,j]
