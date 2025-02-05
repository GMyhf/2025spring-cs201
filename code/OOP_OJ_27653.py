def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

class Fraction:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.simplify()

    def simplify(self):
        common = gcd(self.numerator, self.denominator)
        self.numerator //= common
        self.denominator //= common

    def __add__(self, other):
        numerator = self.numerator * other.denominator + self.denominator * other.numerator
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

a, b, c, d = map(int, input().split())
x = Fraction(a, b)
y = Fraction(c, d)
print(x+y)
