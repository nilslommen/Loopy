Here, we provide the implementation of the algorithm introduced in our paper "On Deciding Constant Runtime of Linear Loops".
For all algebraic computations, we use SymPy (https://www.sympy.org/).

## Setting up the Environment
Python must be installed in order for the implementation to be executed.

To install SymPy, an environment can be set up in following way:
```
python3 -m venv sympy-env
source sympy-env/bin/activate
pip install sympy
```

Afterwards, this environment can always be activated by `source sympy-env/bin/activate` and deactivated by running `deactivate` in the bash.

## Running our Tool
The tool requires the input to be provided in matrix form.
For example, our leading example from paper can be represented by
```
[
    [],                     # C1 (strict linear part of the guard)
    [],                     # c1 (strict affine part of the guard)
    [[1, 1], [-1, -1]],     # C2 (inclusive linear part of the guard)
    [0, 10],                # c2 (inclusive affine part of the guard)
    [[1, 0], [0, 2]],       # A  (linear part of the update)
    [1, 0]                  # b  (affine part of the update)
]
```
This is also stored in the file `examples\example01.koat`.
Now, our tool can be executed by the following command:

```
python src/constant_runtime.py examples/example01.matrix
```

or if you are also interested in more detailed output by

```
python src/constant_runtime.py --log examples/example01.matrix
```

Finally, we provide a tool that transforms a simplified variant of the `*.koat` format, as used in TermComp, into the matrix representation (see below for a more detailed description).
Here, `python src/its2matrix.py examples/example01.koat` prints the matrix representation to the standard output whereas `python src/its2matrix.py -o output.matrix examples/example01.koat` stores the result in the file `output.matrix`.

The transition system below yields the matrix representation shown above.
```
(GOAL COMPLEXITY)
(STARTTERM (FUNCTIONSYMBOLS l0))
(VAR x1 x2)
(transitionS
    l0(x1,x2) -> l1(x1,x2)
    l1(x1,x2) -> l1(x1 + 1,2*x2) :|: 0 <= x1 + x2 && x1 + x2 <= 10
)
```

Note that we allow arbitrary rational numbers as coefficients and restricted the transition system format to represent single-path loops in the following way:
- The input must contain **exactly two transitions**.
- One transition is an **initial transition**: It must start at the declared STARTTERM and move directly to the self-loop's location. The initial transition must **not** contain updates or guards.
- The other transition is the **self-loop**: It is a transition whose stays at the same location, and it may contain affine updates and an optional guard. **Only this self-loop is transformed** into matrices.
