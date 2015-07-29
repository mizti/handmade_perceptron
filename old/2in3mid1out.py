# coding: UTF-8
import numpy as np
#import matplotlib.pyplot as plt
import sys

def judge(wvec, xvec):
    if (np.dot(wvec, xvec) > 0):
        return 1
    else:
        return 0

def train2(wvec, xvec, result, label):
    low = 0.3
    if (result == 1 and label == 1):
        print "pattern 1"
        return wvec
    elif (result == 1 and label == 0):
        wvec_new = wvec - low*xvec
        print "pattern 2"
        print "wvec_new="
        print wvec_new
        return wvec_new
    elif (result == 0 and label == 1):
        wvec_new = wvec + low*xvec
        print "pattern 3"
        print "wvec_new="
        print wvec_new
        return wvec_new
    elif (result == 0 and label == 0):
        print "pattern 4"
        return wvec
    else:
        print "error"

if __name__ == '__main__':
    #データ数
    train_num = 100
    #class1
    x1_1 = np.random.rand(train_num/2) * 5 + 1    #例: [X, Y, Z, ...]
    x1_2 = np.random.rand(train_num/2) * 5 + 1
    label_x1 = np.ones(train_num/2) #クラス1である、というラベル(全て1)
    
    #class2
    x2_1 = (np.random.rand(train_num/2) * 5 + 1 ) * -1
    x2_2 = (np.random.rand(train_num/2) * 5 + 1 ) * -1
    label_x2 = np.zeros(train_num/2) #クラス2である、というラベル(全て0)

    x0 = np.ones(train_num/2)
    x1 = np.c_[x0, x1_1, x1_2]
    x2 = np.c_[x0, x2_1, x2_2]

    xvecs = np.r_[x1, x2] # データを一体化
    xvecs[49][1] = -1
    print xvecs
    labels = np.r_[label_x1, label_x2] # [1, 1, 1, ... , 1, 0, 0, ... 0] 計100データ分の教師データ

    wvecs = np.array([[2, -1, 3],[-1, -1, 1],[3, -2, 4], [2, 2, 2]]) #初期重みベクトル

    for j in range(xvecs[:,0].size):
        print "j=" + str(j)
        #中間層の出力
        y0 = judge(wvecs[0],xvecs[j])
        y1 = judge(wvecs[1],xvecs[j])
        y2 = judge(wvecs[2],xvecs[j])
        yvec = np.array([y0, y1, y2])

        #出力層の出力
        result = judge(wvecs[3],yvec)
        print "result" + str(result)
        print "label" + str(labels[j])

        #labelとresultが一致しなかったら各パーセプトロンを訓練
        if(labels[j] != result):
            wvecs[0] = train2(wvecs[0], xvecs[j], yvec[0], result)
            wvecs[1] = train2(wvecs[1], xvecs[j], yvec[1], result)
            wvecs[2] = train2(wvecs[2], xvecs[j], yvec[2], result)
            wvecs[3] = train2(wvecs[3], yvec, result, labels[j])

#    plt.scatter(x1[:,1], x1[:,2], c='red', marker="o")
#    plt.scatter(x2[:,1], x2[:,2], c='yellow', marker="o")
#    #分離境界線
#    x_fig = np.array(range(-8,8))
#    y_fig = -(wvec[3][1]/wvec[3][2])*x_fig - (wvec[3][0]/wvec[3][2])
#    plt.plot(x_fig,y_fig)
#    plt.show()

