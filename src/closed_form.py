import sympy as sp

from typing import List, Tuple, Optional
from enum import Enum, auto

class InequationType(Enum):
    STRICT = auto()
    INCLUSIVE = auto()

    def __str__(self):
        if self is InequationType.STRICT:
            return " > 0"
        elif self is InequationType.INCLUSIVE:
            return " >= 0"
        return self.name

Var = sp.Symbol
Summand = Tuple[sp.Expr, Optional[Var], int, sp.Expr]
PolyExponential = List[Summand]
Atom = Tuple[PolyExponential, InequationType]
Constraint = List[Atom]
Vars = List[Var]

Matrix = sp.Matrix
Vector = sp.Matrix

def closed_form_to_string(pe: PolyExponential, x0: Vars, N0:int = 0) -> str:
    n = sp.symbols("n", integer=True)

    row_strs = []
    for var_of_closed_form, row in zip(x0, pe):
        grouped = {}
        for coeff, var, a, lam in row:
            key = (a, sp.simplify(lam))
            if var is None:
                term_expr = coeff
            else:
                term_expr = coeff * var
            grouped[key] = grouped.get(key, 0) + term_expr

        sorted_keys = sorted(grouped.keys(), key=lambda k: (sp.nsimplify(k[1]), k[0]), reverse=True)

        expr = 0
        for a, lam in sorted_keys:
            coeff_expr = sp.simplify(grouped[(a, lam)])
            if coeff_expr == 0:
                continue
            expr += coeff_expr * (n**a) * (lam**n)

        row_strs.append(f"{var_of_closed_form}: {sp.simplify(expr)}")

    return "[" + "; ".join(row_strs) + "]" + f" # N0 = {N0}"


def closed_form_linear(A: Matrix, x0: Vars) -> Tuple[List[PolyExponential], bool, int]:
    """
    Computes closed form for the affine iteration
        x_{n+1} = A x_n + b
    where A is a square rational matrix of dimension d and x0 is a column vector of (sympy) symbols
    """

    # n: The symbolic variable representing the index.
    n = sp.symbols('n', integer=True)

    # Compute Jordan form A = P * J * P^{-1}.
    P, J = A.jordan_form()
    P_inv = P.inv()
    d = A.shape[0]
    negativ_eigenvalue = False

    # Compute information about every Jordan block of J:
    #  - start_index: where the block begins in J
    #  - m: block size
    #  - lam: eigenvalue
    #  - N: the m x m nilpotent (super-diagonal) matrix with ones on the super-diagonal
    blocks_info = []
    idx = 0
    while idx < d:
        lam = sp.simplify(J[idx, idx])
        if lam.is_real is False:
            raise ValueError(f"Expected real eigenvalues, got {lam} which is not real.")
        if lam.is_real and lam.is_negative:
            negativ_eigenvalue = True

        # compute block size
        m = 1
        while idx + m < d and J[idx + m - 1, idx + m] == 1:
            m += 1

        blocks_info.append((idx, m, lam))
        idx += m

    # N0 = maximum size among blocks with eigenvalue 0 (they vanish for n >= N0).
    N0 = max((m for _, m, lam in blocks_info if lam == 0), default=0)


    J_entries = [[[] for _ in range(d)] for _ in range(d)]

    # Compute J^n via https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix#Jordan_decomposition
    for (start, m, lam) in blocks_info:
        for i in range(m):
            for j in range(m):
                r = j - i
                if r < 0 or r >= m:
                    J_entries[start + i][start + j] = []
                else:

                    # Polynomial binomial(n,r) expanded in n
                    if r == 0:
                        poly = sp.Integer(1)
                    else:
                        poly = sp.prod([n - k for k in range(r)]) / sp.factorial(r)

                    poly = sp.Poly(poly, n)
                    deg = poly.degree()

                    # Summand has the form (c_s * lam^{-r}, s, lam) representing (c_s * lam^{-r}) * n^s * lam^n
                    summands = []
                    if lam != 0:
                        # For lam != 0: extract coefficients c_s of n^s in binomial(n,r)
                        for s in range(0, deg + 1):
                            c_s = sp.simplify(poly.coeff_monomial(n**s))
                            # multiply by lam^{-r} so the term becomes (c_s * lam^{-r}) * n^s * lam^n
                            coeff = sp.simplify(c_s * (lam ** (-r)))
                            if coeff != 0:
                                summands.append((coeff, s, lam))
                    J_entries[start + i][start + j] = summands

    # Scale summands by a scalar
    def scale_summands(summands, scalar):
        if scalar == 0:
            return []
        else:
            return [(sp.simplify(coeff * scalar), e, lam_val) for (coeff, e, lam_val) in summands if sp.simplify(coeff * scalar) != 0]

    # Compute M = P * J^n
    M_entries = [[[] for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(d):
            acc = []
            for k in range(d):
                p_scalar = sp.simplify(P[i, k])
                if p_scalar != 0:
                    # scale J_entries[k][j] by p_scalar
                    acc = acc + scale_summands(J_entries[k][j], p_scalar)
            M_entries[i][j] = acc

    # Compute A^n = (P * J^n) * P_inv
    A_entries = [[[] for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(d):
            acc = []
            for k in range(d):
                pinv_scalar = sp.simplify(P_inv[k, j])
                if pinv_scalar != 0:
                    # scale J_entries[i][k] by pinv_scalar
                    acc = acc + scale_summands(M_entries[i][k], pinv_scalar)
            A_entries[i][j] = acc

    # Compute row_summands: A^n * x0
    row_summands = [
        [
            (coeff, x0[j], e, lam)
            for j in range(d)
            for (coeff, e, lam) in A_entries[i][j]
            if coeff != 0
        ]
        for i in range(d)
    ]

    return row_summands, negativ_eigenvalue, int(N0)

def closed_form_affine(A: Matrix, b: Vector, x0: Vars) -> Tuple[List[PolyExponential], bool, int]:
    """
    Computes closed form for the affine iteration
        x_{n+1} = A x_n + b
    where A is a square rational matrix of dimension d, b is a rational column vector, and x0 is a column vector of (sympy) symbols
    """

    d = A.shape[0]

    # Make A x_n + b homogenous: [A b; 0 1]
    A_hom = sp.zeros(d + 1)
    for i in range(d):
        for j in range(d):
            A_hom[i, j] = sp.simplify(A[i, j])
    for i in range(d):
        A_hom[i, d] = sp.simplify(b[i, 0])
    A_hom[d, d] = sp.Integer(1)

    x_fresh = sp.Symbol("x_fresh")
    x0_hom = list(x0) + [x_fresh]

    row_summands_hom, negativ_eigenvalue, N0 = closed_form_linear(A_hom, x0_hom)

    # Drop the last row (corresponding to x_fresh)
    row_summands_hom = row_summands_hom[:d]

    row_summands_hom = [
        [(coeff, None if var == x_fresh else var, e, lam) for (coeff, var, e, lam) in row]
        for row in row_summands_hom
    ]

    return row_summands_hom, negativ_eigenvalue, int(N0)

def shift_poly_exponential(expr: PolyExponential, x: int, y: int) -> PolyExponential:
    """
    Shift poly-exponential expression by substituting n -> x + n*y.
    """
    n = sp.symbols("n", integer=True)

    shifted_expr = []

    for coeff, var, a, lam in expr:
        n_poly = sp.expand((x + y*n)**a)

        # lam^(x + n*y) = lam^x * (lam^y)^n
        lam_shifted = lam**x
        lam_y = lam**y

        for k in range(a + 1):
            coeff_nk = sp.simplify(n_poly.coeff(n, k))
            if coeff_nk != 0:
                new_coeff = sp.simplify(coeff * lam_shifted * coeff_nk)
                shifted_expr.append((new_coeff, var, k, lam_y))

    return shifted_expr

def constraint_to_string(constraint: Constraint) -> str:

    n = sp.symbols("n", integer=True)

    row_strs = []
    for row in constraint:
        terms, inequationType = row
        grouped = {}
        for coeff, var, a, lam in terms:
            key = (a, sp.simplify(lam))
            if var is None:
                term_expr = coeff
            else:
                term_expr = coeff * var
            grouped[key] = grouped.get(key, 0) + term_expr

        sorted_keys = sorted(grouped.keys(), key=lambda k: (sp.nsimplify(k[1]), k[0]), reverse=True)

        expr = 0
        for a, lam in sorted_keys:
            coeff_expr = sp.simplify(grouped[(a, lam)])
            if coeff_expr == 0:
                continue
            expr += coeff_expr * (n**a) * (lam**n)

        row_strs.append(str(sp.simplify(expr)) + str(inequationType))

    return "[" + "; ".join(row_strs) + "]"

def compute_constraint(C1: Matrix, c1: Vector, C2: Matrix, c2: Vector, A: Matrix, b: Vector, x0: Vars) -> Tuple[Constraint, bool, int]:
    """
    Computes closed form of the guard Ci*(A x_0 + b)^n + ci.
    Here, C1 and c1 represent strict inequations whereas C2 and c2 represent inclusive inequations.
    """
    from logger import log_string

    # Get closed form for x_n as list of per-row terms
    terms_xn, negative_eigenvalue, N0 = closed_form_affine(A, b, x0)
    log_string("closed form: " + closed_form_to_string(terms_xn,x0,N0))

    def projection(C: Matrix, c: Vector, type: InequationType) -> Constraint:
        r, d = C.shape
        all_rows = []

        for i in range(r):
            row_terms = []

            # Project each variable-row term into constraint row
            for row_idx in range(d):
                for coeff, var, a, lam in terms_xn[row_idx]:
                    if var is None:
                        row_terms.append((sp.simplify(C[i, row_idx] * coeff), None, a, lam))
                    else:
                        row_terms.append((sp.simplify(C[i, row_idx] * coeff), var, a, lam))

            # Add affine constant c
            if c[i] != 0:
                row_terms.append((sp.simplify(c[i]), None, 0, sp.Integer(1)))

            # Group by (var, a, lam) to merge coefficients
            grouped = {}
            for coeff, var, a, lam in row_terms:
                key = (var, a, lam)
                grouped[key] = grouped.get(key, 0) + coeff

            final_row_terms = ([
                (sp.simplify(coeff), var, a, lam)
                for (var, a, lam), coeff in grouped.items()
                if coeff != 0
            ], type)
            all_rows.append(final_row_terms)

        return all_rows

    strict_guard = projection(C1, c1, InequationType.STRICT)
    inclusive_guard = projection(C2, c2, InequationType.INCLUSIVE)

    all_rows = strict_guard + inclusive_guard

    log_string("guard: " + constraint_to_string(all_rows))

    return all_rows, negative_eigenvalue, N0
