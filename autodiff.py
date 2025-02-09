import math as m

def ad(f):
    """Automatically differentiate a function with one variable

    Returns a function over x0 which returns tuple (f(x0), f'(x0))"""
    return lambda x0: f(ADN(x0, 1)).to_tuple()

class ADN:
    """Automatic differentiation dual number"""
    def __init__(self, r: float, a: float):
        self.real = r
        self.derived = a

    def __add__(self, other):
        other = raisenum(other)
        return ADN(self.real + other.real, self.derived + other.derived)
    def __radd__(self, other):
        other = raisenum(other)
        return ADN(other.real + self.real, other.derived + self.derived)

    def __sub__(self, other):
        other = raisenum(other)
        return ADN(self.real - other.real, self.derived - other.derived)
    def __rsub__(self, other):
        other = raisenum(other)
        return ADN(other.real - self.real, other.derived - self.derived)
    def __neg__(self):
        return 0.0 - self

    def __mul__(self, other):
        other = raisenum(other)
        return ADN(self.real * other.real, self.derived * other.real + self.real * other.derived)
    def __rmul__(self, other):
        other = raisenum(other)
        return ADN(other.real * self.real, other.derived * self.real + other.real * self.derived)

    def __truediv__(self, other):
        other = raisenum(other)
        return ADN(self.real / other.real, (self.derived * other.real - self.real * other.derived) / (other.real)**2)
    def __rtruediv__(self, other):
        other = raisenum(other)
        return ADN(other.real / self.real, (other.derived * self.real - other.real * self.derived) / (self.real)**2)

    def __pow__(self, other: float):
        return ADN(self.real ** other, self.derived * other * self.real ** (other-1))

    def __rpow__(self, other: float):
        return exp(self * ln(other))

    def __abs__(self):
        return ADN(abs(self.real), self.derived * (lambda x: (0 if x == 0 else ((-1, 1)[x > 0])))(self.real))

    def to_tuple(self):
        """Returns a tuple with the real and derived value i.e. (x, x')"""
        return (self.real, self.derived)

def raisenum(x):
    """Creates an ADN(x, 0) from a constant x"""
    if isinstance(x, float) or isinstance(x, int):
        return ADN(x, 0)
    elif isinstance(x, ADN):
        return x
    else:
        raise TypeError(f"value {x} cannot be raised to dual number")

def sin(x):
    x = raisenum(x)
    return ADN(m.sin(x.real), x.derived * m.cos(x.real))

def cos(x):
    x = raisenum(x)
    return ADN(m.cos(x.real), -(x.derived * m.sin(x.real)))

def tan(x):
    x = raisenum(x)
    return sin(x)/cos(x)

def cot(x):
    x = raisenum(x)
    return cos(x)/sin(x)

def log(x, base=m.e):
    x = raisenum(x)
    return ADN(m.log(x.real, base), x.derived / x.real)

def ln(x):
    return log(x)

def exp(x):
    x = raisenum(x)
    return ADN(m.e ** x.real, x.derived * (m.e ** x.real))
