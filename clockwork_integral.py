import sympy as sp
R, m, y, phi = sp.symbols("R, m, y, phi")

I = 4 * sp.exp(2*m*y) / (1 + sp.exp(2*m*y) * phi)**2
ans = sp.integrate(I, (y, 0, sp.pi*R))
ans = ans.simplify()
print(ans)
