from typing import Tuple, Iterable
from closed_form import Var, Summand, PolyExponential, Atom, Constraint, Vars, InequationType

def rootbound(pe: PolyExponential) -> int:
    """
    Given a list of summands in the form (coeff, e, lam), compute the rootbound:
        M - 1 + sum_{i=1}^M e_i
    where M is the number of terms and e_i are the exponents of n.
    """
    M = sum(1 for _ in {(a, lam) for _, _, a, lam in pe})
    exponent_sum = sum(a for a, _ in {(a, lam) for _, _, a, lam in pe})
    return M - 1 + exponent_sum

def compute_rb(constraint: Constraint) -> int:
    """
    Computes sum_{psi in constraint} rootbound(psi).
    """
    return sum(rootbound(psi) for psi, _ in constraint)

import sympy as sp

def separate_x(atom: Atom, x: Var) -> Tuple[PolyExponential, PolyExponential, InequationType]:
    """
    Rewrites an atom pe ~ 0 into p*x + t ~ 0 where p is variable-free and t does not contain x.
    """
    pe, inequationType = atom

    p_terms = []
    t_terms = []

    for (coeff, var, a, lam) in pe:
        if var == x:
            p_terms.append((coeff, var, a, lam))
        else:
            t_terms.append((coeff, var, a, lam))

    return p_terms, t_terms, inequationType


def esign(p_terms: PolyExponential) -> PolyExponential:
    # Sort lexicographically: Larger base b first, then larger exponent e
    sorted_terms = sorted(p_terms,key=lambda triple: (triple[3], triple[2]), reverse=True)
    if len(sorted_terms) != 0:
        d, _, _, _ = sorted_terms[0]
        if d > 0:
            return 1
        else:
            return -1
    else:
        return 0

def multiply_p_with_t(p_terms: PolyExponential, t_terms: PolyExponential) -> PolyExponential:
    """
    This function computes p * t.
    """
    result = []
    for (coeff, _, a_p, lam_p) in p_terms:
        for (poly, var, a_t, lam_t) in t_terms:
            result.append((coeff * poly, var, a_p + a_t, sp.simplify(lam_p * lam_t)))

    # Group by (var, a, lam) to merge coefficients
    grouped = {}
    for coeff, var, a, lam in result:
        key = (var, a, lam)
        grouped[key] = grouped.get(key, 0) + coeff

    return [(sp.simplify(coeff), var, a, lam) for (var, a, lam), coeff in grouped.items() if coeff != 0]

def negate_t(t_terms: PolyExponential) -> PolyExponential:
    """
    This function computes -t.
    """
    result = []
    for (coeff, var, a, lam) in t_terms:
        result.append((sp.simplify(-1 * coeff), var, a, lam))

    return result

# The following four functions are useful to represent constraints as *sets* of poly-exponential expressions.
def _canonical_summand(t: Summand) -> Summand:
    coeff, var, a, lam = t
    return (sp.simplify(coeff), var, a, sp.simplify(lam))

def _canonical_poly_exponential(pe: PolyExponential) -> Tuple[Summand]:
    # Group by (var, a, lam) to merge coefficients
    grouped = {}
    for coeff, var, a, lam in pe:
        key = (var, a, lam)
        grouped[key] = grouped.get(key, 0) + coeff

    pe = [(sp.simplify(coeff), var, a, lam) for (var, a, lam), coeff in grouped.items() if coeff != 0]

    canonical_summands = [_canonical_summand(t) for t in pe]

    def _sort_key(summand):
        coeff, var, a, lam = summand
        return (str(lam), a, str(var) if var is not None else "", str(coeff))

    canonical_summands.sort(key=_sort_key)
    return tuple(canonical_summands)

def constraint_to_set(constraint: Constraint) -> set:
    return { (_canonical_poly_exponential(inner), inequationType) for (inner, inequationType) in constraint }

def set_to_constraint(s) -> Constraint:
    return [ ([ tuple(t) for t in inner ], inequationType) for (inner, inequationType) in s ]

def eliminate_x(pi: Constraint, x: Var) -> Constraint:
    """
    Eliminates the variable x in the constraint pi in the spirit of Fourier-Motzkin.
    """

    from logger import log_string
    from closed_form import constraint_to_string
    log_string(f"\n\tEliminate {x} in")
    log_string("\t" + constraint_to_string(pi))

    pi_split = []
    for terms in pi:
        pi_split.append(separate_x(terms, x))

    # Partition pi into positive and negative according to sign of highest summand
    pi_pos = [(p,t,inequationType) for (p,t,inequationType) in pi_split if esign(p) == 1]
    pi_neg = [(p,t,inequationType) for (p,t,inequationType) in pi_split if esign(p) == -1]

    pi_res = []

    def combine_inequations(a: InequationType, b: InequationType) -> InequationType:
        if InequationType.STRICT in (a, b):
            return InequationType.STRICT
        return InequationType.INCLUSIVE

    for (p1,t1,ineq1) in pi_pos:
        for (p2,t2,ineq2) in pi_neg:
            pi_res.append((multiply_p_with_t(p1,t2) + negate_t(multiply_p_with_t(p2,t1)), combine_inequations(ineq1,ineq2)))

    pi_set = constraint_to_set(pi)

    # Merge p and t again
    pi_pos = [(p + t,inequationType) for (p,t,inequationType) in pi_pos]
    pi_neg = [(p + t,inequationType) for (p,t,inequationType) in pi_neg]

    pi_pos_set = constraint_to_set(pi_pos)
    pi_neg_set = constraint_to_set(pi_neg)
    pi_res_set = constraint_to_set(pi_res)

    return set_to_constraint((pi_set | pi_res_set) - (pi_pos_set | pi_neg_set))

def eliminate_vars(constraint: Constraint, x0: Vars, N0:int = 0) -> Constraint:
    """
    Eliminates all variables x0 in the constraint pi in the spirit of Fourier-Motzkin.
    """

    from logger import log_string
    rb = compute_rb(constraint)

    log_string(f"\nQuantifier Elimination:\n\trb: {rb}")

    from closed_form import shift_poly_exponential, constraint_to_string
    pi = [(shift_poly_exponential(expr, N0, k), inequationType) for k in range(rb + 1) for (expr, inequationType) in constraint]

    log_string("\tpi: " + constraint_to_string(pi))

    result = pi
    for x in x0:
        result = eliminate_x(result, x)

    return result

def has_constant_runtime(constraint: Constraint, x0: Vars, N0:int = 0):
    constraint_var_free = eliminate_vars(constraint, x0, N0)

    from logger import log_string
    log_string("")
    if any(esign(expr) == -1 or (esign(expr) == 0 and inequationType == InequationType.STRICT)
           for (expr, inequationType) in constraint_var_free):
        print("CONSTANT")
    else:
        print("NON-CONSTANT")
