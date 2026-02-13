import sympy as sp
import re

# -------------------------
# Expects 6 entries [C1, c1, C2, c2, A, b] in the file (comments with # are ignored)
# -------------------------
def read_matrices_from_file(filename):
    """
    Read matrices from a file containing exactly one top-level list with six entries: [C1, c1, C2, c2, A, b].
    Our leading example:

        [
            [],                     # C1 (strict linear part of the guard)
            [],                     # c1 (strict affine part of the guard)
            [[1, 1], [-1, -1]],     # C2 (inclusive linear part of the guard)
            [0, 10],                # c2 (inclusive affine part of the guard)
            [[1, 0], [0, 2]],       # A  (linear part of the update)
            [1, 0]                  # b  (affine part of the update)
        ]

    Fractions like `1/2` are parsed as sympy.Rational(1,2).
    """

    with open(filename, "r") as f:
        content = f.read()

    # Remove comments (anything after '#') and trim whitespace/newlines
    content = re.sub(r"#.*", "", content).strip()

    # Parse the cleaned string using sympy
    parsed = sp.sympify(content, evaluate=False)

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 6:
        raise ValueError("File must contain exactly four entries [C1, c1, C2, c2, A, b].")

    # Helper: Convert a nested list into a sympy.Matrix with rational entries
    def to_matrix(obj):

        if not isinstance(obj, (list, tuple)):
            raise ValueError("Each entry must be a list (matrix or flat list).")

        first = obj[0] if len(obj) > 0 else None
        first_is_row = isinstance(first, (list, tuple)) or getattr(first, "is_List", False)
        if not first_is_row:
            return sp.Matrix([[sp.sympify(str(e), rational=True)] for e in obj])

        return sp.Matrix([[sp.sympify(str(e), rational=True) for e in row] for row in obj])

    C1 = to_matrix(parsed[0])
    c1 = to_matrix(parsed[1])
    C2 = to_matrix(parsed[2])
    c2 = to_matrix(parsed[3])
    A = to_matrix(parsed[4])
    b = to_matrix(parsed[5])

    # Basic sanity checks on dimensions
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError("A must be square (d x d).")
    if d == 0:
        if b.shape != (0, 0) and b.shape != (0, 1):
            raise ValueError(f"b must be empty or 0x1 when d=0 (got {b.shape}).")
    else:
        if b.shape != (d, 1):
            raise ValueError(f"b must be d x 1 (got {b.shape}, expected {(d,1)}).")
    if C1.rows > 0 and C1.cols > 0:
        if C1.shape[1] != d:
            raise ValueError(f"C1 must have d columns (got {C1.shape[1]}, expected {d}).")
        if c1.shape[0] != C1.shape[0] or c1.shape[1] != 1:
            raise ValueError("c1 must be a column vector with same number of rows as C1.")
    if C2.rows > 0 and C2.cols > 0:
        if C2.shape[1] != d:
            raise ValueError(f"C2 must have d columns (got {C2.shape[1]}, expected {d}).")
        if c2.shape[0] != C2.shape[0] or c2.shape[1] != 1:
            raise ValueError("c2 must be a column vector with same number of rows as C2.")

    return C1, c1, C2, c2, A, b
