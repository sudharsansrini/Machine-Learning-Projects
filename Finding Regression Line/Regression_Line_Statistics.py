# formula -
import numpy as np
# x = [for i in input().split()]

n = int(input())

x = []

y = []

print("Enter X")
for i in range(n):
    p = float(input())
    x.append(p)

print("Enter Y")
for i in range(n):
    q = float(input())
    y.append(q)

sum_x = []
sum_y = []

x = np.array(x)
y = np.array(y)


for i in range(len(x)):
   sum_x.append(x[i] - np.mean(x))


for i in range(len(y)):
   sum_y.append(y[i] - np.mean(y))

print("X Details")
print("Mean of X : %.2f" % np.mean(x))
sum_x = np.array(sum_x)
print("(X-Xbar) :", sum_x)
print("(X-Xbar)^2: ", sum_x ** 2)
print("summation of (X-Xbar)^2: %.2f" % np.sum(sum_x ** 2))
num_x = np.sum(sum_x ** 2)
Vx = num_x / (n-1)
print("Variance of x: %.2f" % Vx)
sd_x = Vx ** 0.5
print("Standard Deviation of x : %.2f"% sd_x)

print("Y Details")
print("Mean of Y : %.2f"% np.mean(y))
sum_y = np.array(sum_y)
print("(Y-Ybar): ", sum_y)
print("(Y-Ybar)^2: ", sum_y ** 2)
print("summation of (Y-Ybar)^2: %.2f"% np.sum(sum_y ** 2))
num_y = np.sum(sum_y ** 2)
Vy = num_y / (n-1)
print("Variance of y: %.2f"% Vy)
sd_y = Vy ** 0.5
print("Standard Deviation of y : %.2f"% sd_y)

print(sum_x * sum_y)
coV = np.sum(sum_x * sum_y)/(n-1)
print("Co variance of x and y:", coV)

r = coV / (sd_x * sd_y)
print("Correlation Co efficient : %.2f" % r)