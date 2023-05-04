import copy
import random
from random import randint
import time

# random.seed(10)
class Dice:
  value = 0

  def roll(self):
    self.value = randint(1,6)




li = [Dice(), Dice(),Dice(), Dice(), Dice(), Dice()]

for i in li:
  i.roll()

print([i.value for i in li] )




def combs(l):

  ones = []
  twos = []
  threes = []
  fours = []
  fives = []
  sixs = []

  for a in range(len(l)):
    ones.append([l[a]])
    for b in range(len(l)):
      if b > a: twos.append([l[a],l[b]])
      for c in range(len(l)):
        if c > b > a: threes.append([l[a],l[b],l[c]])
        for d in range(len(l)):
          if d > c > b > a: fours.append([l[a],l[b],l[c],l[d]])
          for e in range(len(l)):
            if e > d > c > b > a: fives.append([l[a],l[b],l[c],l[d],l[e]])
            for f in range(len(l)):
              if f > e > d > c > b > a: sixs.append([l[a],l[b],l[c],l[d],l[e],l[f]])

  return [ones, twos, threes, fours, fives, sixs]


all = combs(li)

all = all
all_val = []

for i in all:
  f = []
  for j in i:
    v = []
    for jj in j:
      v.append(jj.value)

    f.append(v)

  all_val.append(f)

for i in all_val:
  for j in i:
    print(j)

z = []
zz = []
print("**********")
for c in range(4,13):
  t = copy.copy(all)
  t_val = copy.copy(all_val)
  y = []
  x = []
  for i in range(len(t)):
    for j in range(len(t[i])):
      if sum(t_val[i][j]) == c:
        y.append(t[i][j])
        x.append(t_val[i][j])


  z.append(y)
  zz.append(x)



for i in zz:
  print(i)
print("*********'")

to_sum = []
for i in z:
  s = []
  for j in i:
    s.append(j)
    for dice in j:
      for jj in i:
        if dice in jj and jj != j:
          i.remove(jj)
          # print("rem", [jjj.value for jjj in jj])


zz = []
for i in z:
  t = []
  for j in i:
    val = []
    t.append([jj.value for jj in j])
  zz.append(t)

for i in zz:
  print(i)

sums = []

s = 0
for i in li:

  if i.value <= 3:
    s+=i.value

sums.append(s)
for i in zz:
  s = 0
  for j in i:
    s+=sum(j)
  sums.append(s)
print(sums)
print(max(sums))
