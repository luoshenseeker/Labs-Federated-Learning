import matplotlib.pyplot as plt
import numpy as np
from math import exp
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def func(x,y):  # 下降目标函数
    res = exp((-0.1)*((x*x)+(y*y)))
    return res

def d_func(x,y):
    dx = -0.2*x*exp((-0.1)*((x*x)+(y*y)))
    dy = -0.2*y*exp((-0.1)*((x*x)+(y*y)))
    return dx,dy

def gradient_descent(x_start, y_start, epochs, learning_rate):
    theta_x = []
    theta_y = []
    temp_x = x_start
    temp_y = y_start

    theta_x.append(temp_x)
    theta_y.append(temp_y)

    for i in range(epochs):
        dx,dy = d_func(temp_x, temp_y)
        temp_x = temp_x - dx*learning_rate
        temp_y = temp_y - dy*learning_rate

        theta_x.append(temp_x)
        theta_y.append(temp_y)

    return theta_x[::3], theta_y[::3]

# 绘制3D箭头构造的函数
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

# 画图函数
# meshgrid网格的用法:把两个向量(m维,n维)之间可以组成的配对方式进行全排列，生成两个n行*m列的矩阵，每个矩阵的下标有一一对应关系
def mat_plot(epochs=12,learning_rate=0.8):
    delta = 0.025
    a = np.arange(-4, 4, delta)
    b = np.arange(-4, 4, delta)
    x, y = np.meshgrid(a, b)
    z = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x)):
            r = func(x[i][j],y[i][j])
            temp.append(r)
        z.append(temp)
    z =  np.array(z)
    dx_cen, dy_cen = gradient_descent(-0.9, -0.9, epochs, learning_rate)  # 获取梯度下降原方向和步长
    dx_cen[4] += -0.2
    dy_cen[4] += -0.2
    dz_cen = []
    dz1 = []
    dz2 = []
    dz3 = []
    dz4 = []
    for i in range(len(dx_cen)):
        dz_cen.append(func(dx_cen[i],dy_cen[i]))

    # 1、设置几组偏离数据(稀疏型)
    dx1 = [-0.9,-0.8,-0.9,-1.2,-1.68]
    dy1 = [-0.9, -1.64, -2.30, -2.88, -3.48]
    dx2 = [-0.9,-0.40,0,0.2,0]
    dy2 = [-0.9,-1.78,-2.49,-3.24,-4.0]
    dx3 = [-0.9, -1.64, -2.30, -2.88, -3.46]
    dy3 = [-0.9,-0.8,-0.9,-1.2,-1.52]
    dx4 = [-0.9,-1.78,-2.39,-3.08,-3.78]
    dy4 = [-0.9,-0.40,0,0.2,0]
    # 2、设置几组偏离数据(紧缩型)
    # dx1 = [-0.9,-0.98,-1.3,-1.65,-2.18]
    # dy1 = [-0.9, -1.50, -2.05, -2.57, -3.10]
    # dx2 = [-0.9,-0.65,-0.60,-0.95,-1.4]
    # dy2 = [-0.9,-1.68,-2.40,-2.95,-3.5]
    # dx3 = [-0.9, -1.50, -2.05, -2.57, -3.10]
    # dy3 = [-0.9,-0.98,-1.3,-1.65,-2.18]
    # dx4 = [-0.9,-1.68,-2.40,-2.95,-3.5]
    # dy4 = [-0.9,-0.65,-0.60,-0.95,-1.4]

    for i in range(len(dx_cen)):
        dz1.append(func(dx1[i],dy1[i]))
        dz2.append(func(dx2[i],dy2[i]))
        dz3.append(func(dx3[i],dy3[i]))
        dz4.append(func(dx4[i],dy4[i]))

    # 画3D模型图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dx[::3], dy[::3], dz[::3], color='r', s=10,alpha=1)  # 散点图绘制
    # ax.plot(dx[::3], dy[::3], dz[::3], linewidth=1, linestyle='-',color='r')  # 直线绘制
    # ax.contour(x,y,z)    # 简体图绘制
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('GnBu'), alpha=0.7)
    # ax.contourf(x, y, z, zdir='z', offset=-2) # 画二维投影图
    ax.view_init(elev=45., azim=210)  # 视角1:设置为转换视角：elev=90为仰角状态，azim为方位角状态
    # 绘制箭头
    for i in range(len(dx_cen)-1):
        a = Arrow3D([dx_cen[i], dx_cen[i+1]], [dy_cen[i], dy_cen[i+1]], [dz_cen[i], dz_cen[i+1]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r", alpha=1)
        ax.add_artist(a)
        b = Arrow3D([dx1[i], dx1[i+1]], [dy1[i], dy1[i+1]], [dz1[i], dz1[i+1]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="b", alpha=1)
        ax.add_artist(b)
        c = Arrow3D([dx2[i], dx2[i + 1]], [dy2[i], dy2[i + 1]], [dz2[i], dz2[i + 1]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="g", alpha=1)
        ax.add_artist(c)
        d = Arrow3D([dx3[i], dx3[i + 1]], [dy3[i], dy3[i + 1]], [dz3[i], dz3[i + 1]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="b", alpha=1)
        ax.add_artist(d)
        e = Arrow3D([dx4[i], dx4[i + 1]], [dy4[i], dy4[i + 1]], [dz4[i], dz4[i + 1]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="g", alpha=1)
        ax.add_artist(e)
    plt.title("low_variance")
    plt.savefig("gradient_low.pdf", format='pdf', dpi=1000, bbox_inches="tight")
    plt.show()

    # # 画2D局部图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # """
    # 箭头起始位置（A[0],A[1]）和终点位置（B[0],B[1]）
    # length_includes_head = True:表示增加的长度包含箭头部分
    # head_width:箭头的宽度
    # head_length:箭头的长度
    # fc:filling color(箭头填充的颜色)
    # ec:edge color(边框颜色)
    # """
    # for i in range(len(dx_cen)-1):
    #     ax.arrow(dx_cen[i], dy_cen[i], dx_cen[i+1]-dx_cen[i], dy_cen[i+1]-dy_cen[i],
    #              length_includes_head=True, head_width=0.1, head_length=0.1, fc='r', ec='r')
    #     ax.arrow(dx1[i], dy1[i], dx1[i + 1] - dx1[i], dy1[i + 1] - dy1[i],
    #              length_includes_head=True, head_width=0.1, head_length=0.1, fc='b', ec='b')
    #     ax.arrow(dx2[i], dy2[i], dx2[i + 1] - dx2[i], dy2[i + 1] - dy2[i],
    #              length_includes_head=True, head_width=0.1, head_length=0.1, fc='g', ec='g')
    #     ax.arrow(dx3[i], dy3[i], dx3[i + 1] - dx3[i], dy3[i + 1] - dy3[i],
    #              length_includes_head=True, head_width=0.1, head_length=0.1, fc='b', ec='b')
    #     ax.arrow(dx4[i], dy4[i], dx4[i + 1] - dx4[i], dy4[i + 1] - dy4[i],
    #              length_includes_head=True, head_width=0.1, head_length=0.1, fc='g', ec='g')
    #     ax.set_xlim(-4, 1)  # 设置图形的范围，默认为[0,1]
    #     ax.set_ylim(-4, 1)  # 设置图形的范围，默认为[0,1]
    #     ax.set_aspect('equal')  # x轴和y轴等比例
    #
    # plt.axis('off')
    # plt.show()

mat_plot()
