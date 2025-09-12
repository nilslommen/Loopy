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


def closed_form_affine(A: Matrix, b: Vector, x0: Vars) -> Tuple[List[PolyExponential], int]:
    """
    Computes closed form for the affine iteration
        x_{n+1} = A x_n + b
    where A is a square rational matrix of dimension d, b is a rational column vector, and x0 is a column vector of (sympy) symbols
    """

    # n: The symbolic variable representing the index.
    # k: A dummy summation index (used only when evaluating finite sums like sum_{k=0}^{n-1} ...)
    n, k = sp.symbols('n k', integer=True)

    # Compute Jordan form A = P * J * P^{-1}.
    P, J = A.jordan_form()
    P_inv = P.inv()
    size = A.shape[0]

    # Compute information about every Jordan block of J:
    #  - start_index: where the block begins in J
    #  - m: block size
    #  - lam: eigenvalue
    #  - N: the m x m nilpotent (super-diagonal) matrix with ones on the super-diagonal
    blocks_info = []
    idx = 0
    while idx < size:
        lam = sp.simplify(J[idx, idx])
        if lam.is_real is False:
            raise ValueError(f"Expected real eigenvalues, got {lam} which is not real.")

        # compute block size
        m = 1
        while idx + m < size and J[idx + m - 1, idx + m] == 1:
            m += 1
        # build nilpotent matrix
        N = sp.zeros(m)
        for i in range(m - 1):
            N[i, i + 1] = 1
        blocks_info.append((idx, m, lam, N))
        idx += m

    # N0 = maximum size among blocks with eigenvalue 0 (they vanish for n >= N0).
    N0 = max((m for _, m, lam, _ in blocks_info if lam == 0), default=0)

    # Collect list of tuples (lam, t, M) where M is a full-size matrix representing lam^{-t} * N^t.
    # This will be helpful for J^n = sum_{t = 0}^{m - 1} binomial(n, t) * lam^{n - t} * N^t.
    M_terms = []
    for start, m, lam, N in blocks_info:
        if lam == 0:
            # skip pure-zero blocks: they are nilpotent and vanish for k >= N0
            continue
        for t in range(m):
            # lam^{-t} * N^t  (identity when t == 0)
            if t == 0:
                submatrix = sp.eye(m)
            else:
                submatrix = sp.simplify(lam**(-t) * (N**t))
            # lift sub-matrices to full-size matrices
            full_matrix = sp.zeros(size)
            for i in range(m):
                for j in range(m):
                    full_matrix[start + i, start + j] = submatrix[i, j]
            # conjugate back to original basis
            M = sp.simplify(P * full_matrix * P_inv)
            M_terms.append((lam, t, M))

    # Compute contributions from "linear" A^n x0 and from "affine" sum_{k=0}^{n-1} A^k b part
    summands = []

    for lam, t, M in M_terms:

        # "linear" A^n x0 part:
        M_vars = sp.simplify(M * x0)

        # binomial(n, t) is a polynomial in n of degree t
        binom_expr = sp.expand(sp.binomial(n, t))
        for i in range(0, t + 1):                        # coefficients possible for degrees 0,...,t
            coeff = sp.simplify(binom_expr.coeff(n, i))  # coefficient of n^i
            if coeff != 0:
                summands.append((sp.simplify(coeff * M_vars), i, lam))


        # "affine" sum_{k=0}^{n-1} A^k b part:
        coeff_vec_b = sp.simplify(M * b)
        if coeff_vec_b == sp.zeros(size, 1):
            continue

        if sp.simplify(lam - 1) == 0:
            # Case lam == 1: sum_{k = 0}^{n - 1} binomial(k, t) = binomial(n, t + 1)

            # binomial(n, t + 1) is a polynomial in n of degree t + 1
            binom_expr = sp.simplify(sp.binomial(n, t + 1))
            for i in range(0, t + 2):                       # coefficients possible for degrees 0,...,t + 1
                coeff = sp.simplify(binom_expr.coeff(n, i)) # coefficient of n^i
                if coeff != 0:
                    summands.append((sp.simplify(coeff * coeff_vec_b), i, sp.Integer(1)))
        else:
            # Case lam != 1: Compute closed-form for sum_{k = 0}^{n - 1} binomial(k, t) * lam^k

            # Tries to compute closed form expression of sum_{k = 0}^{n - 1} binomial(k, t) * lam^k
            # https://docs.sympy.org/latest/modules/concrete.html#sympy.concrete.summations.summation
            S = sp.simplify(sp.summation(sp.binomial(k, t) * lam**k, (k, 0, n - 1)))

            # Expand closed form:
            # https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html#expand
            S_expanded = sp.simplify(sp.expand(S))

            # Try to isolate the lam^n term: S_expanded = coeff_n(n) * lam^n + remainder
            coeff_n = sp.simplify(S_expanded.coeff(lam**n))

            if coeff_n != 0:

                for a in range(0, t + 2):
                    c = sp.simplify(coeff_n.coeff(n, a))
                    if c != 0:
                        summands.append((sp.simplify(c * coeff_vec_b), a, lam))

                # remainder = S - coeff_n * lam^n
                remainder = sp.simplify(S - coeff_n * lam**n)
                if remainder != 0:
                    # store remainder (i.e., multiplied by 1^n)
                    summands.append((sp.simplify(remainder * coeff_vec_b), 0, sp.Integer(1)))
            else:
                raise ValueError(
                    f"Failed to extract lam^n term for eigenvalue lam={lam} with block size {t}."
                    f"SymPy returned closed form {S_expanded}."
                )


    row_summands = [[] for _ in range(size)]

    for coeff_vec, a, lam in summands:
        for r in range(size):
            expr = sp.simplify(coeff_vec[r])
            if expr == 0:
                continue
            vars_in_expr = expr.free_symbols & set(x0)
            if not vars_in_expr:
                row_summands[r].append((sp.simplify(expr), None, a, lam))
            else:
                var = vars_in_expr.pop()
                coeff = sp.simplify(expr.coeff(var))
                row_summands[r].append((coeff, var, a, lam))

    return row_summands, int(N0)

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

def compute_constraint(C1: Matrix, c1: Vector, C2: Matrix, c2: Vector, A: Matrix, b: Vector, x0: Vars) -> Tuple[Constraint, int]:
    """
    Computes closed form of the guard Ci*(A x_0 + b)^n + ci.
    Here, C1 and c1 represent strict inequations whereas C2 and c2 represent inclusive inequations.
    """
    from logger import log_string

    # Get closed form for x_n as list of per-row terms
    terms_xn, N0 = closed_form_affine(A, b, x0)
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

    return all_rows, N0
