from itertools import product

alph = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

i = 0

d= 0

_j = ''.join
step = 0
for c in product(alph, repeat=8):
    step += 1
    if step % 10000000 == 0:
        print(step)
    s = _j(c)
    if 'бак' in s and 'куб' in s: # ответ
        d+=1

print(d)