"""
Simplified *.koat -> Matrix Representation
--------------------------------------------------

For the transition system representation

    (GOAL COMPLEXITY)
    (STARTTERM (FUNCTIONSYMBOLS l0))
    (VAR x1 x2)
    (RULES
        l0(x1,x2) -> l1(x1,x2)
        l1(x1,x2) -> l1(x1 + 1,2*x2) :|: 0 <= x1 + x2 && x1 + x2 <= 10
    )

we obtain the matrix representation

    [
        [],                     # C1 (strict linear part of the guard)
        [],                     # c1 (strict affine part of the guard)
        [[1, 1], [-1, -1]],     # C2 (inclusive linear part of the guard)
        [0, 10],                # c2 (inclusive affine part of the guard)
        [[1, 0], [0, 2]],       # A  (linear part of the update)
        [1, 0]                  # b  (affine part of the update)
    ]

Note that we allow arbitrary rational numbers as coefficients (where rational numbers can be written as fractions or in decimal notation). Moreover, we restricted the transition system format to represent single-path loops in the following way:
- The input must contain **exactly two transitions**.
- One transition is an **initial transition**: It must start at the declared STARTTERM and move directly to the self-loop's location. The initial transition must **not** contain non-identity updates or guards.
- The other transition is the **self-loop**: It is a transition that stays in the same location, and it may contain affine updates and an optional guard. **Only this self-loop is transformed** into the matrix representation.
"""

import argparse
import re
import sympy as sp

def parse_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    # remove comments (from '#' to end of line)
    text = re.sub(r"#.*", "", text)
    return text


def extract_vars(text):
    m = re.search(r"\(VAR\s+([^\)]*)\)", text)
    if not m:
        return []
    return m.group(1).split()

def is_self_loop(transition):
    """
    Returns true if the transition is a self-loop: l(...) -> l(...)
    """
    parts = transition.split("->", 1)
    if len(parts) < 2:
        return False
    lhs, rhs_guard = parts

    # strip arguments and spaces
    lhs_name = lhs.split("(")[0].strip()
    rhs_name = rhs_guard.split("(")[0].strip()

    return lhs_name == rhs_name

def validate_initial_transition(transition):
    """
    Raises an error if the initial transition has any updates or guards.
    Acceptable form: f(x1,...,xn) -> g(x1,...,xn) with no guard
    """
    if ":|:" in transition:
        raise ValueError(f"Initial transition cannot have a guard: {transition}")

    parts = transition.split("->", 1)
    lhs, rhs = parts

    # Check if RHS arguments are exactly the same as LHS
    lhs_args = re.search(r"\((.*)\)", lhs)
    rhs_args = re.search(r"\((.*)\)", rhs)
    if rhs_args is not None and rhs_args.group(1).strip():

        lhs_args_str = lhs_args.group(1).strip() if lhs_args else ""
        rhs_args_str = rhs_args.group(1).strip()
        if lhs_args_str != rhs_args_str:
            raise ValueError(f"Initial transition cannot update arguments: {transition}")

def extract_transitions(text):
    """Extract transitions inside (RULES ... ) by balancing parentheses"""

    start = text.find("(RULES")
    if start == -1:
        return []
    depth = 0
    end = None

    for i, char in enumerate(text[start:], start=start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError("Unbalanced parentheses in RULES block")

    # Content between "(RULES" and the matching ")"
    transitions_block = text[start + len("(RULES"):end].strip()
    transitions = [r.strip() for r in transitions_block.splitlines() if r.strip()]
    return transitions


def parse_affine(expr_str, variables):
    """
    Parse an affine expression and return a pair: (list of coefficients, constant)
    """
    local_dict = {str(v): v for v in variables}

    expr = sp.sympify(expr_str, rational=True, locals=local_dict)
    expr = sp.expand(expr)

    if variables:
        p = sp.Poly(expr, variables)
        if p.total_degree() > 1:
            raise ValueError(f"Expression is not affine: {expr_str}")

        coeffs = [sp.Rational(expr.coeff(v)) for v in variables]
        const = sp.Rational(expr.subs({v: 0 for v in variables}))
    else:
        coeffs = []
        const = sp.Rational(expr)

    return coeffs, const


def parse_transition(transition, variables):
    """
    Parse a single transition of the form:
        l1(x1,...,xn) -> l1(a1,...,an)  or  l1(x1,...,xn) -> l1(a1,...,an) :|: guard
    Return matrix representation [C1, c1, C2, c2, A, b] of the given transition.
    """
    parts = transition.split("->", 1)
    rhs_guard = parts[1].strip()

    if ":|:" in rhs_guard:
        rhs, guard = rhs_guard.split(":|:", 1)
        guard = guard.strip()
    else:
        rhs, guard = rhs_guard, None

    # Parse rhs args: Find the parentheses in rhs
    m = re.search(r"\((.*)\)", rhs)
    if not m:
        raise ValueError("Invalid RHS arguments in transition: " + transition)
    rhs_args_str = m.group(1).strip()
    rhs_args = [] if rhs_args_str == "" else [a.strip() for a in rhs_args_str.split(",")]

    # Build A, b from rhs affine polynomials
    A, b = [], []
    for arg in rhs_args:
        coeffs, const = parse_affine(arg, variables)
        A.append([c for c in coeffs])
        b.append(const)

    # Parse guard (split by &&)
    C1, c1, C2, c2 = [], [], [], []
    if guard:
        def normalize_guard(cond, variables):

          cond = cond.strip()
          if ">=" in cond:
              left, right = cond.split(">=", 1)
              strict = False
          elif ">" in cond:
              left, right = cond.split(">", 1)
              strict = True
          elif "<=" in cond:
              left, right = cond.split("<=", 1)
              # flip sides: right - left >= 0
              left, right = right, left
              strict = False
          elif "<" in cond:
              left, right = cond.split("<", 1)
              left, right = right, left
              strict = True
          else:
              raise ValueError(f"Guard condition must contain one of '>', '>=', '<', '<=': {cond}")

          expr = sp.sympify(left, rational=True) - sp.sympify(right, rational=True)
          expr = sp.expand(expr)
          coeffs, const = parse_affine(expr, variables)
          return coeffs, const, strict

        conditions = [g.strip() for g in guard.split("&&") if g.strip()]
        for cond in conditions:
            coeffs, const, strict = normalize_guard(cond, variables)
            if strict:
                C1.append([c for c in coeffs])
                c1.append(const)
            else:
                C2.append([c for c in coeffs])
                c2.append(const)


    return [C1, c1, C2, c2, A, b]


def transform(filename):
    text = parse_file(filename)
    var_names = extract_vars(text)
    variables = [sp.Symbol(v) for v in var_names]

    # Parse start location
    m = re.search(r"\(STARTTERM\s*\(\s*FUNCTIONSYMBOLS\s+(\w+)\s*\)\s*\)", text)
    if not m:
        raise ValueError("Cannot find start location.")
    start_location_name = m.group(1).strip()

    # Extract transitions
    transitions = extract_transitions(text)
    if len(transitions) != 2:
        raise ValueError(f"System must have exactly two transitions, found {len(transitions)}")

    # Identify self-loop: must NOT start from start location
    self_loops = [
        r for r in transitions
        if is_self_loop(r) and r.split("->")[0].split("(")[0].strip() != start_location_name
    ]
    if len(self_loops) != 1:
        raise ValueError(
            f"System must have exactly one self-loop (not starting from start location), found {len(self_loops)}"
        )
    self_loop_transition = self_loops[0]
    self_loop_name = self_loop_transition.split("->")[0].split("(")[0].strip()

    # Identify initial transition: must start from start location
    initial_transitions = [
        r for r in transitions
        if r.split("->")[0].split("(")[0].strip() == start_location_name
    ]
    if len(initial_transitions) != 1:
        raise ValueError(
            f"There must be exactly one initial transition starting from the start location, found {len(initial_transitions)}"
        )
    initial_transition = initial_transitions[0]

    # Validate initial transition (no updates or guards)
    validate_initial_transition(initial_transition)

    # Check that initial transition ends in the self-loop
    initial_rhs_name = initial_transition.split("->")[1].split("(")[0].strip()
    if initial_rhs_name != self_loop_name:
        raise ValueError(
            f"Initial transition must go from start location '{start_location_name}' directly to the self-loop '{self_loop_name}', "
            f"but it goes to '{initial_rhs_name}'"
        )

    # Transform the self-loop transition
    return parse_transition(self_loop_transition, variables)


def to_string(v):
    if isinstance(v, list):
        return "[" + ", ".join(to_string(x) for x in v) + "]"
    else:
        return str(sp.simplify(v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform from transition system (*.koat) to matrix representation.")
    parser.add_argument("input", help="Input file (e.g., input.koat)")
    parser.add_argument("-o", "--output", help="Output file (if omitted, prints to stdout)")

    args = parser.parse_args()

    result = transform(args.input)

    output_lines = []
    output_lines.append("[")
    for i, row in enumerate(result):
        comma = "," if i < len(result) - 1 else ""
        output_lines.append("  " + to_string(row) + comma)
    output_lines.append("]")

    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
    else:
        print(output_text)
