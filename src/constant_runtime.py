import sympy as sp
import argparse
from logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input file (e.g., example.matrix)")
    parser.add_argument("--log", action="store_true", help="Enables logging output")
    args = parser.parse_args()

    if not args.log:
        logger.disabled = True

    from read_matrices import read_matrices_from_file
    C1, c1, C2, c2, A, b = read_matrices_from_file(args.input)

    # Build symbolic initial vector x0 = (x1,...,xd)^T
    d = A.shape[0]
    x0 = sp.Matrix(sp.symbols(f'x1:{d+1}'))

    from closed_form import compute_constraint
    terms_proj, N0 = compute_constraint(C1, c1, C2, c2, A, b, x0)

    from quantifier_elimination import has_constant_runtime
    has_constant_runtime(terms_proj, x0, N0)
