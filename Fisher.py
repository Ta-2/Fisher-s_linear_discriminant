import numpy as np
import matplotlib.pyplot as plt

#年齢データ
age_datas = np.array(
    [25.0, 35.0, 70.0, 50.0, 30.0, 20.0, 40.0]
    )

#性別データ　（男=1, 女=-1）
gender_datas = np.array(
    [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    )

#購入データ　（購入する=1.0, 購入しない=-1.0）
buy_datas = np.array(
    [1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1,.0]
    )

datas = np.array([
    np.array([age, gen, buy])
    for age, gen, buy in zip(age_datas, gender_datas, buy_datas)
    ])

def devide_group(datas, tag):
    cnt = 0
    group = []
    m = [0.0, 0.0]
    for data in filter(lambda data: data[2] == tag, datas):
        cnt += 1
        group.append( [data[0], data[1]] )
        m[0] += data[0]
        m[1] += data[1]
    m[0] /= cnt
    m[1] /= cnt

    return np.array(group), np.array(m)

group1, m1 = devide_group(datas, 1.0)
group2, m2 = devide_group(datas, -1.0)

def Var_Cov_Matrix(datas, mean):
    Var_Cov = np.array([0.0, 0.0]).reshape(1, 2)
    Var_Cov = Var_Cov.T.dot(Var_Cov)

    for data in datas:
        vec = (data - mean).reshape(1, 2)
        Var_Cov += vec.T.dot(vec)
    
    return Var_Cov

Sw = Var_Cov_Matrix(group1, m1)
Sw += Var_Cov_Matrix(group2, m2)
print("Sw: ")
print(Sw)

m21 = (m2 - m1).reshape(1,2)
Sb = m21.T.dot(m21)
print("Sb: ")
print(Sb)

Sw_inv = np.linalg.inv(Sw)
w = Sw_inv.dot((m2-m1).T.reshape(2,1))
#length = np.linalg.norm(w)
#w /= length
print("w: ")
print(w)
#w0 = -1/2 * w.dot(m1 + m2)
x = -w[1]
y = w[0]
y /= x
a = w[0]/-w[1]
m_c = (m1 + m2)/2.0
b = m_c[1] - a*m_c[0]
print("a, b: ")
print(a, b)

def plot_vec(x, y, u, v, c):
    plt.quiver(x, y, u, v, angles='xy',scale_units='xy',scale=1, color=c, width=0.005)

plt.scatter(x=[d[0] for d in group1], y=[d[1] for d in group1], c="red")
plt.scatter(x=[d[0] for d in group2], y=[d[1] for d in group2], c="blue")

plt.plot([0, 80.0], [b, a*80.0+b], c="green")

#plot_vec(0, 0, m1[0], m1[1], "red")
#plot_vec(0, 0, m2[0], m2[1], "blue")

plt.xlim([0,80])
plt.ylim([-4,4])
plt.grid()
plt.show()

age_test_data = [20.0, 25.0, 45.0, 50.0, 60.0, 60.0, 70.0]
gen_test_data = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]

for age, gen in zip(age_test_data, gen_test_data):
    print("age: {}, gen: {}, buy: {}".format(age, gen, "true" if age*a+b>0.0 else "false"))


plt.scatter(x=age_test_data, y=gen_test_data, c="black")
plt.scatter(x=[d[0] for d in group1], y=[d[1] for d in group1], c="red")
plt.scatter(x=[d[0] for d in group2], y=[d[1] for d in group2], c="blue")

plt.plot([0, 80.0], [b, a*80.0+b], c="green")

plt.xlim([0,80])
plt.ylim([-4,4])
plt.grid()
plt.show()

Vw = w.T.dot(Sw.dot(w))
Vb = w.T.dot(Sb.dot(w))
print("Vw: {}, Vb: {}, eval: {}".format(Vw, Vb, Vb/Vw))

delta_eval = (2*Sb.dot(w)*Vw - 2*Sw.dot(w)*Vb)/(Vw**2)
print("delta_eval: {}".format(delta_eval))