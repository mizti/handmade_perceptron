# -*- coding: utf-8 -*-

class test_class:

def __init__(self, code, name):
self.code = code
self.name = name

# neuronクラス作成されると、3つの入力に対して掛けられる
# 3つの初期重みw1~w3が設定される。
# また、judgeメソッドは3つの値を入力すると0もしくは1を出力する
class neuron:

def __init__(self, w0, w1, w2):
self.w0 = w0
self.w1 = w1
self.w2 = w2

def judge(self, x1, x2, x3):
v = self.w0 * x1 + self.w1 * x2 + self.w2 * x3
if v > 0:
return 1
else:
return 0

if __name__ == "__main__":

# 入力層のニューロンを3つ定義する(x0=1)
x0 = 1 #これは固定値
x1 = 3
x2 = 0.5
t = 1
# 中間層のニューロンを3つ定義する
h1 = neuron(-1,0.8,1)
a1 = h1.judge(x0, x1, x2)

h2 = neuron(-1,1,1)
a2 = h2.judge(x0, x1, x2)

# 出力層のニューロンを1つ定義する
a0 = 1 #これは固定値
theta = neuron(-1,1,1)
y = theta.judge(a0, a1, a2)
print "y=" + str(y)

epsilon = 0.01

# ??これでいいの？
# w0も触るの?
# なぜa1掛けるの? -> a1が正で t=1 > y=0ならw1を減らすべき:a1が負で t=1 >
y=0ならw1を増やすべきという増減方向を調整するため
# thetaのw0~w2も触るの? -> 触るんじゃないかな...?

# if t != y:
# h1.w0 += epsilon * (t-y) * a1 # W10
# h1.w1 += epsilon * (t-y) * a1 # W11
# h1.w2 += epsilon * (t-y) * a1 # W12
# h2.w0 += epsilon * (t-y) * a2 # W20
# h2.w1 += epsilon * (t-y) * a2 # W21
# h2.w2 += epsilon * (t-y) * a2 # W22
