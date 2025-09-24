import numpy as np
from src.utils import candidates, ind_constraint, cum31, cum22, independence


def TestCase(T, O, Z):
    if ind_constraint(T, O, Z)[0]:
        return 'a'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(Z, T, O - root1 * T)
    flag2, _ = ind_constraint(Z, T, O - root2 * T)
    if flag1 ^ flag2:
        return 'b'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(O - root1 * T, Z, T)
    flag2, _ = ind_constraint(O - root2 * T, Z, T)
    if flag1 or flag2:
        return 'c'
    
    root1, root2 = candidates(O, Z)
    flag1, _ = ind_constraint(T, O, Z - root1 * O)
    flag2, _ = ind_constraint(T, O, Z - root2 * O)
    if flag1 or flag2:
        return 'd'
    
    root_t1, root_t2 = candidates(Z, T)
    root_o1, root_o2 = candidates(Z, O)
    flag1, _ = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
    flag2, _ = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
    flag3, _ = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
    flag4, _ = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
    if flag1 or flag2 or flag3 or flag4:
        return 'e'

    return 'f,g,h'


def MulTestCase(T, O, Z):
    if ind_constraint(T, O, Z)[0]:
        return 'a'

    root1, root2 = candidates(Z, O)
    flag1, _ = ind_constraint(O - root1 * Z, T, Z)
    flag2, _ = ind_constraint(O - root2 * Z, T, Z)
    if flag1 or flag2:
        return 'c'
    
    root1, root2 = candidates(O, Z)
    flag1, _ = ind_constraint(T, O, Z - root1 * O)
    flag2, _ = ind_constraint(T, O, Z - root2 * O)
    if flag1 or flag2:
        return 'd'

    root1, root2 = candidates(T, Z)
    flag1, _ = ind_constraint(T, O, Z - root1 * T)
    flag2, _ = ind_constraint(T, O, Z - root2 * T)
    if flag1 or flag2:
        return 'f'

    root_t1, root_t2 = candidates(Z, T)
    root_o1, root_o2 = candidates(Z, O)
    flag1, _ = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
    flag2, _ = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
    flag3, _ = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
    flag4, _ = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
    if flag1 or flag2 or flag3 or flag4:
        return 'b,e'
    
    root_t1, root_t2 = candidates(T, Z)
    Z1, Z2 = Z - root_t1 * T, Z - root_t2 * T
    root_z11, root_z12 = candidates(Z1, O)
    root_z21, root_z22 = candidates(Z2, O)
    flag1, _ = ind_constraint(T, O - root_z11 * Z1, Z1)
    flag2, _ = ind_constraint(T, O - root_z12 * Z1, Z1)
    flag3, _ = ind_constraint(T, O - root_z21 * Z2, Z2)
    flag4, _ = ind_constraint(T, O - root_z22 * Z2, Z2)
    if flag1 or flag2 or flag3 or flag4:
        return 'g'

    else:
        return 'h'


def GenTestCase(T, O, Z):
    flag1, _ = independence(T - np.cov(T,Z)[0,1] / np.cov(T,Z)[1,1] * Z, Z)
    flag2, _ = independence(Z - np.cov(Z,T)[0,1] / np.cov(Z,T)[1,1] * T, T)
    flag3, _ = independence(Z - np.cov(Z,O)[0,1] / np.cov(Z,O)[1,1] * O, O)
    coeff = np.linalg.inv(np.cov(T,O)) @ np.array([np.cov(T,Z)[0,1], np.cov(O,Z)[0,1]])
    flag4, _ = independence(Z - coeff[0]*T - coeff[1]*O,T)
    if flag1 or flag2 or flag3 or flag4:
        return 'a,b,c,d,e,f,g,h,i,j,k,l'
    return 'm,n,o,p,q,r,s,t'


def CalCase(T, O, Z, case):
    
    def perfect(T, O, Z):
        if cum31(T, Z) / cum31(Z, T) > 0:
            ratio1 = np.sign(np.cov(T,Z)[0,1]) * np.sqrt(cum31(T, Z) / cum31(Z, T))
            ratio2 = cum31(T, Z) / cum22(T, Z)
            ratio3 = cum22(T, Z) / cum31(Z, T)
            ratio = np.median([ratio1, ratio2, ratio3])
        else:
            ratio1 = cum31(T, Z) / cum22(T, Z)
            ratio2 = cum22(T, Z) / cum31(Z, T)
            ratio = (ratio1 + ratio2) / 2
        result = (np.cov(T,O)[0,1] - ratio * np.cov(O,Z)[0,1]) / (np.var(T) - ratio * np.cov(T,Z)[0,1])
        return result

    if case == 'a':
        return perfect(T, O, Z)
    if case == 'b':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(Z, T, O - root1 * T)
        _, p2 = ind_constraint(Z, T, O - root2 * T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'c':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(O - root1 * T, Z, T)
        _, p2 = ind_constraint(O - root2 * T, Z, T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'd':
        root1, root2 = candidates(O, Z)
        _, p1 = ind_constraint(T, O, Z - root1 * O)
        _, p2 = ind_constraint(T, O, Z - root2 * O)
        if p1 > p2:
            return perfect(T, O, Z - root1 * O)
        else:
            return perfect(T, O, Z - root2 * O)
    if case == 'e':
        root_t1, root_t2 = candidates(Z, T)
        root_o1, root_o2 = candidates(Z, O)
        _, p1 = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
        _, p2 = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
        _, p3 = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
        _, p4 = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
        if p1 > max([p2, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o1 * Z, Z)
        elif p2 > max([p1, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o2 * Z, Z)
        elif p3 > max([p1, p2, p4]):
            return perfect(T - root_t2 * Z, O - root_o1 * Z, Z)
        else:
            return perfect(T - root_t2 * Z, O - root_o2 * Z, Z)
    else:
        raise ValueError


def CalComplexCase(T, O, Z1, Z2, case):
    
    def double_perfect(T, O, Z1, Z2):
        if cum31(T, Z1) / cum31(Z1, T) > 0:
            ratio11 = np.sign(np.cov(T,Z1)[0,1]) * np.sqrt(cum31(T, Z1) / cum31(Z1, T))
            ratio12 = cum31(T, Z1) / cum22(T, Z1)
            ratio13 = cum22(T, Z1) / cum31(Z1, T)
            ratio1 = np.median([ratio11, ratio12, ratio13])
        else:
            ratio11 = cum31(T, Z1) / cum22(T, Z1)
            ratio12 = cum22(T, Z1) / cum31(Z1, T)
            ratio1 = (ratio11 + ratio12) / 2

        if cum31(T, Z2) / cum31(Z2, T) > 0:
            ratio21 = np.sign(np.cov(T,Z2)[0,1]) * np.sqrt(cum31(T, Z2) / cum31(Z2, T))
            ratio22 = cum31(T, Z2) / cum22(T, Z2)
            ratio23 = cum22(T, Z2) / cum31(Z2, T)
            ratio2 = np.median([ratio21, ratio22, ratio23])
        else:
            ratio21 = cum31(T, Z2) / cum22(T, Z2)
            ratio22 = cum22(T, Z2) / cum31(Z2, T)
            ratio2 = (ratio21 + ratio22) / 2

        result = (np.cov(T,O)[0,1] - ratio1 * np.cov(O,Z1)[0,1] - ratio2 * np.cov(O,Z2)[0,1]) / (np.var(T) - ratio1 * np.cov(T,Z1)[0,1] - ratio2 * np.cov(T,Z2)[0,1])
        return result
    
    if case == 'a':
        return double_perfect(T, O, Z1, Z2)
    
    if case == 'b':
        root_t1, root_t2 = candidates(Z1, T)
        root_o1, root_o2 = candidates(Z1, O)
        _, p1 = ind_constraint(T - root_t1 * Z1, O - root_o1 * Z1, Z1)
        _, p2 = ind_constraint(T - root_t1 * Z1, O - root_o2 * Z1, Z1)
        _, p3 = ind_constraint(T - root_t2 * Z1, O - root_o1 * Z1, Z1)
        _, p4 = ind_constraint(T - root_t2 * Z1, O - root_o2 * Z1, Z1)
        if p1 > max([p2, p3, p4]):
            lamda1, lamda2 = root_t1, root_o1
        elif p2 > max([p1, p3, p4]):
            lamda1, lamda2 = root_t1, root_o2
        elif p3 > max([p1, p2, p4]):
            lamda1, lamda2 = root_t2, root_o1
        else:
            lamda1, lamda2 = root_t2, root_o2
        root1, root2 = candidates(O, Z2)
        _, p1 = ind_constraint(T, O, Z2 - root1 * O)
        _, p2 = ind_constraint(T, O, Z2 - root2 * O)
        if p1 > p2:
            lamda = root1
        else:
            lamda = root2

        Z2 = Z2 - lamda * O
        T = T - lamda1 * Z1
        O = O - lamda2 * Z1
        return double_perfect(T, O, Z1, Z2)

    if case == 'c':
        root1, root2 = candidates(T, Z1)
        _, p1 = ind_constraint(T, O, Z1 - root1 * T)
        _, p2 = ind_constraint(T, O, Z1 - root2 * T)
        if p1 > p2:
            lamda = root1
        else:
            lamda = root2
        root_t1, root_t2 = candidates(Z2, T)
        root_o1, root_o2 = candidates(Z2, O)
        _, p1 = ind_constraint(T - root_t1 * Z2, O - root_o1 * Z2, Z2)
        _, p2 = ind_constraint(T - root_t1 * Z2, O - root_o2 * Z2, Z2)
        _, p3 = ind_constraint(T - root_t2 * Z2, O - root_o1 * Z2, Z2)
        _, p4 = ind_constraint(T - root_t2 * Z2, O - root_o2 * Z2, Z2)
        if p1 > max([p2, p3, p4]):
            lamda1, lamda2 = root_t1, root_o1
        elif p2 > max([p1, p3, p4]):
            lamda1, lamda2 = root_t1, root_o2
        elif p3 > max([p1, p2, p4]):
            lamda1, lamda2 = root_t2, root_o1
        else:
            lamda1, lamda2 = root_t2, root_o2

        Z1 = Z1 - lamda * T
        T = T - lamda1 * Z2
        O = O - lamda2 * Z2
        return double_perfect(T, O, Z1, Z2)
    
    else:
        raise ValueError