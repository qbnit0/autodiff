# Autodiff
Autodiff is a simple proof-of-concept for automatic differentiation in Python.
It provides a class for "dual numbers" which overload mathematical operators to automatically differentiate any elementary function.
For functions from Python's math library (sin, cos, log, etc.), autodiff provides drop-in replacements that work with dual numbers.