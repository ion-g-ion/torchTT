import torchtt as tntt
import torch as tn

A = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1])
B = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 3, 2, 2, 1])

Cr = A @ B

C = tntt.amen_mm(A, B, kickrank=8, verbose=False)


print((C-Cr).norm()/Cr.norm())

A = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1])
B = tntt.random([(4, 3), (5, 2), (6, 5), (3, 6)], [1, 3, 2, 2, 1])

Cr = A @ B

C = tntt.amen_mm(A, B, kickrank=8, verbose=False)


print((C-Cr).norm()/Cr.norm())