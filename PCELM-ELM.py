# coding: utf-8
import sys
from numpy import arange, where, array, dot, outer, zeros, concatenate, ones, tile, mean, trace, diag
import numpy as np
from numpy.linalg import inv, norm, svd
from scipy.io import loadmat

def KennardStone(X, Num):
    nrow = X.shape[0]
    CalInd = zeros((Num), dtype=int)-1
    vAll = arange(0, nrow)
    D = zeros((nrow, nrow))
    for i in range(nrow-1):
        for j in range(i+1, nrow):
            D[i, j] = norm(X[i, :]-X[j, :])
    ind = where(D == D.max())
    CalInd[0] = ind[1]
    CalInd[1] = ind[0]
    for i in range(2, Num):
        vNotSelected = array(list(set(vAll)-set(CalInd)))
        vMinDistance = zeros(nrow-i)
        for j in range(nrow-i):
            nIndexNotSelected = vNotSelected[j]
            vDistanceNew = zeros((i))
            for k in range(i):
                nIndexSelected = CalInd[k]
                if nIndexSelected <= nIndexNotSelected:
                    vDistanceNew[k] = D[nIndexSelected,nIndexNotSelected]
                else:
                    vDistanceNew[k] = D[nIndexNotSelected, nIndexSelected]
            vMinDistance[j] = vDistanceNew.min()
        nIndexvMinDistance = where(vMinDistance == vMinDistance.max())
        CalInd[i] = vNotSelected[nIndexvMinDistance]
    ValInd = array(list(set(vAll)-set(CalInd)))
    return CalInd, ValInd

def plscvfold(X, y, A, K):
    sort_index = np.argsort(y, axis = 0)
    y = np.sort(y, axis = 0)
    X = X[sort_index[:, 0]]
    M = X.shape[0]
    yytest = zeros([M, 1])
    YR = zeros([M, A])
    groups = np.asarray([i % K + 1 for i in range(0, M)])
    group = np.arange(1, K+1)
    for i in group:
        Xtest = X[groups == i]
        ytest = y[groups == i]
        Xcal = X[groups != i]
        ycal = y[groups != i]
        index_Xtest = np.nonzero(groups == i)
        index_Xcal = np.nonzero(groups != i)

        (Xs, Xp1, Xp2) = pretreat(Xcal)
        (ys, yp1, yp2) = pretreat(ycal)
        PLS1 = pls1_nipals(Xs, ys, A)
        W, T, P, Q = PLS1['W'],  PLS1['T'],  PLS1['P'], PLS1['Q']
        yp = zeros([ytest.shape[0], A])
        for j in range(1, A+1):
            B = dot(W[:, 0:j], Q.T[0:j])
            C = dot(B, yp2) / Xp2
            coef = concatenate((C, yp1-dot(C.T, Xp1)), axis = 0)
            Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis = 1)
            ypred = dot(Xteste, coef)
            yp[:, j-1:j] = ypred

        YR[index_Xtest, :] = yp
        yytest[index_Xtest, :] = ytest
        #print("The %sth group finished" %i )

    error =YR - tile(y, A)
    errs = error * error
    PRESS = np.sum(errs, axis=0)
    RMSECV_ALL = np.sqrt(PRESS/M)
    index_A = np.nonzero(RMSECV_ALL == min(RMSECV_ALL))
    RMSECV_MIN = min(RMSECV_ALL)
    SST = np.sum((yytest - mean(y))**2)
    Q2_all = 1-PRESS/SST
    return {'index_A': index_A[0] + 1, 'RMSECV_ALL': RMSECV_ALL, 'Q2_all': Q2_all}

def pls1_nipals(X, y, a):
    T = zeros((X.shape[0], a))
    P = zeros((X.shape[1], a))
    Q = zeros((1, a))
    W = zeros((X.shape[1], a))
    for i in range(a):
        v = dot(X.T, y[:, 0])
        
        W[:, i] = v/norm(v)
        T[:, i] = dot(X, W[:, i])
        P[:, i] = dot(X.T, T[:, i])/dot(T[:, i].T, T[:, i])
        Q[0, i] = dot(T[:, i].T, y[:, 0])/dot(T[:, i].T, T[:, i])
        X = X-outer(T[:, i], P[:, i])
    W = dot(W, inv(dot(P.T, W)))
    B = dot(W[:, 0:a], Q[:, 0:a].T)
    return {'B': B, 'T': T, 'P': P, 'Q': Q, 'W': W}

def plspredtest(B, Xtest, xp1, xp2, yp1, yp2):
    C = dot(B, yp2) / xp2
    coef = concatenate((C, yp1-dot(C.T, xp1)), axis = 0)
    Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis = 1)
    ypred = dot(Xteste, coef)
    return ypred

def RMSEP(ypred, Ytest):
    error = ypred - Ytest
    errs = error ** 2
    PRESS = np.sum(errs)
    RMSEP = np.sqrt(PRESS/Ytest.shape[0])
    SST = np.sum((Ytest - np.mean(Ytest))**2)
    Q2 = 1-PRESS/SST
    return RMSEP, Q2

    
def error(pre, signal):
    err = pre - signal
    err = err * err
    return err.sum()


def pretreat(X):
    [M, N] = X.shape
    p1 =np.mean(X, axis=0).reshape(N, 1)
    p2 = np.ones([N, 1])
    Xs = np.zeros([M, N])
    for i in range(0, N):
        Xs[:, i:i+1] = ((X[:, i:i+1] - p1[i])/p2[i])
    return Xs, p1, p2

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

##  Extreme Learning Machine AutoEncoder##
class ELMAutoEncoder(object):
    """
    Extreme Learning Machine Auto Encoder :
        __init__ :
            activation : Layer's activation
            n_hidden : Hidden Layer's number of neuron
            coef : coefficient for Layer's ridge redression
            seed : seed for np.random.RandomState
            domain : domain for initial value of weight and bias
    """

    def __init__(self, activation=sigmoid,
                 n_hidden=50, coef=0.,  seed=123, domain=[-1., 1.]):
        self.activation = activation
        self.n_hidden = n_hidden
        self.coef = coef
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain

    def get_weight(self):
        return self.weight

    def get_bias(self):
         return self.bias

    def get_beta(self):
        return self.layer.beta

    def fit(self, input, signal):
        # set parameter of layer
        self.input = input
        self.n_input = len(input[0])
        self.n_output = len(input[0])
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # orthogonal weight and forcely regularization

        for i in range(len(weight)):
            w = weight[i]
            for j in range(0,i):
                w = w - weight[j].dot(w) * weight[j]
            w = w / np.linalg.norm(w)
            weight[i] = w


        # bias regularization
        denom = np.linalg.norm(bias)
        if denom != 0:
            denom = bias / denom

        # generate self weight and bias
        self.weight = weight
        self.bias = bias


        # generate self layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        # fit layer
        self.layer.fit(input, signal)

    def predict(self, input):
        # get predict_output
        predict_output = []
        predict_hidden = []
        for i in input:
            o = self.layer.get_output(i)[1].tolist()
            h = self.layer.get_output(i)[0].tolist()
            predict_output.append(o)
            predict_hidden.append(h)
        np.save('output.npy',predict_output)
        np.save('hidden.npy',predict_hidden)
        return predict_output

    def score(self, input, teacher):
        # get score
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in range(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length

    def error(self, input, signal):
        # get error
        pre = self.predict(input)
        err = pre - signal
        err = err * err
        return err.sum()

##  Layer  ##
class Layer(object):
    """
    Layer : used for Extreme Learning Machine
        __init__ :
            activation : activation from input to hidden
            n_{input, hidden, output} : each layer's number of neuron
            c : coefficient for ridge regression
            w : weight from input to hidden layer
            b : bias from input to hidden layer
            beta : beta from hidden to output layer
    """
    def __init__(self, activation, size, w, b, c):
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.c = c
        self.w = w
        self.b = b
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])

    def get_beta(self):
        return self.beta

    def get_i2h(self, input):
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        hidden = self.get_i2h(input)  # from input to hidden
        
        output = self.get_h2o(hidden) # from hidden to output

        return hidden,output

    def fit(self, input, signal):
        # get activation of hidden layer
        H = []
        for i, d in enumerate(input):
            sys.stdout.write("\r    input %d" % (i+1))
            sys.stdout.flush()
            H.append(self.get_i2h(d))

        print (" done.")

        # coefficient of regularization
        sys.stdout.write("\r    coefficient")
        sys.stdout.flush()
        np_id = np.identity(min(np.array(H).shape))
        if self.c == 0:
            coefficient = 0
        else:
            coefficient = 1. / self.c
        print (" done.")

        # pseudo inverse
        sys.stdout.write("\r    pseudo inverse")
        sys.stdout.flush()
        H = np.array(H)
        regular = coefficient * np_id
        if H.shape[0] < H.shape[1]:
            Hp = np.linalg.inv(np.dot(H, H.T) + regular)
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.linalg.inv(np.dot(H.T, H) + regular)
            Hp = np.dot(Hp, H.T)
        print (" done.")
        
        # set beta
        sys.stdout.write("\r    set beta")
        sys.stdout.flush()
        self.beta = np.dot(Hp, np.array(signal))
        np.save('beta.npy',self.beta)
        print (" done.")



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import datetime

    datafile = 'cornmat.mat'
    dataset = loadmat(datafile)
    MX = dataset['m5']; SX = dataset['mp5']; Y = dataset['pro'];
    
    
    
    KS_PLS_num = 64 #for pls model
    KS_tran_num = 40 #for cal model
 

#  PLS for Quantification    
    CalInd, ValInd = KennardStone(MX, KS_PLS_num)
    MXcal = MX[CalInd]; MXtest = MX[ValInd]
    SXcal = SX[CalInd]; SXtest = SX[ValInd]
    Ycal = Y[CalInd]; Ytest = Y[ValInd]

 
    OUTPUT = plscvfold(MXcal, Ycal, 15, 10)
    index_A, RMSECV_ALL = OUTPUT['index_A'], OUTPUT['RMSECV_ALL']
    [X, xp1, xp2] = pretreat(MXcal)
    [y, yp1, yp2] = pretreat(Ycal)
    PLS = pls1_nipals(X, y, index_A[0])
    coef = PLS['B']
    Mypred = plspredtest(coef, MXtest, xp1, xp2, yp1, yp2)
    Stestypred = plspredtest(coef, SXtest, xp1, xp2, yp1, yp2)

  
    
    RMSEP_M, Q2_M = RMSEP(Mypred, Ytest)
    RMSEP_S, Q2_S = RMSEP(Stestypred, Ytest)
    

    W = PLS['W']; P = PLS['P']
    TPLS = dot(W,inv(dot(P.T, W)))


#  PCELM for TRANSFER
    CalInd, ValInd = KennardStone(MXcal, KS_tran_num)

    MXtrain = MXcal[CalInd]; MXpre = MXcal[ValInd]
    SXtrain = SXcal[CalInd]; SXpre = SXcal[ValInd]
    Ytrain = Ycal[CalInd]; Ypre = Ycal[ValInd]

    
    MXtrain1 =  MXtrain - np.mean( MXtrain, axis=0)
    SXtrain1 =  SXtrain - np.mean( SXtrain, axis=0)
    MXtest1 =  MXtest - np.mean( MXtrain, axis=0)  
    SXtest1 =  SXtest - np.mean( SXtrain, axis=0)


    TMX = dot(MXtrain1, TPLS)
    
    hiddens = np.arange(200,500,100)
    pre_tests = []; pre_trains =[]; 
    for i, num in enumerate(hiddens):

        ae = ELMAutoEncoder(activation=tanh, n_hidden=num, coef=50000, domain=[-0.1, 0.1])
        ae.fit(SXtrain1, TMX)

        pre_train = ae.predict(SXtrain1)
        pre_trains.append(pre_train)
 
        pre_test = ae.predict(SXtest1)
        pre_tests.append(pre_test)

    SXtestpres = np.asarray(pre_tests)
    SXtestpre1 = np.mean(SXtestpres, axis=0)
    
    SXtestpre = dot(SXtestpre1, P.T)
    SXtestpre =  SXtestpre + np.mean( MXtrain, axis=0)
    err_pre = error(SXtestpre, MXtest)
    Stestpre_ypred = plspredtest(coef, SXtestpre, xp1, xp2, yp1, yp2)
    RMSEP_Stestpre, Q2_Stestpre = RMSEP(Stestpre_ypred, Ytest)
    
    print( "RMSEP_Spre:", RMSEP_Stestpre, "Q2_Spre:", Q2_Stestpre)
    print( "RMSEP_M:", RMSEP_M, "Q2_mp5:", Q2_M)
    print( "RMSEP_S:", RMSEP_S, "Q2_m5:", Q2_S)

    
    [rX, rxp1, rxp2] = pretreat(SXcal)
    [ry, ryp1, ryp2] = pretreat(Ycal)
    rPLS = pls1_nipals(rX, ry, index_A[0])
    rcoef = rPLS['B']
    rMypred = plspredtest(rcoef, SXtest, rxp1, rxp2, ryp1, ryp2)

    rRMSEP_M, rQ2_M = RMSEP(rMypred, Ytest)
    print( "rRMSEP_M:", rRMSEP_M, "rQ2_mp5:", rQ2_M)
    


#  ELM for Quantification
    hiddens_Y = np.arange(200,500,100)
    pre_tests_Y = []; err_tests_Y = [] 
    pre_trains_Y =[]; err_trains_Y =[]
    pre_trans_Y = []; err_trans_Y = []

    for i, num in enumerate(hiddens_Y):
       ae1 = ELMAutoEncoder(activation=tanh, n_hidden=num, coef=50000, domain=[-0.1, 0.1])
       ae1.fit(MXcal, Ycal)

       pre_train_Y = ae1.predict(MXcal)
       pre_trains_Y.append(pre_train_Y)

       pre_test_Y = ae1.predict(MXtest)
       pre_tests_Y.append(pre_test_Y)

       pre_tran_Y = ae1.predict(SXtestpre)
       pre_trans_Y.append(pre_tran_Y)       

    
    pre_tests_Y = np.asarray(pre_tests_Y)
    pre_test_Y = np.mean(pre_tests_Y, axis=0)

    pre_trains_Y = np.asarray(pre_trains_Y)
    pre_train_Y = np.mean(pre_trains_Y, axis=0)
    
    pre_trans_Y = np.asarray(pre_trans_Y)
    pre_tran_Y = np.mean(pre_trans_Y, axis=0)
    
    RMSEP_testpre_Y, Q2_testpre_Y = RMSEP(pre_test_Y, Ytest)
    RMSEP_train_Y, Q2_train_Y = RMSEP(pre_train_Y, Ycal)
    RMSEP_tran_Y, Q2_tran_Y = RMSEP(pre_tran_Y, Ytest)
    print( "RMSEP_M_Y:", RMSEP_train_Y, "Q2_M_Y:", Q2_train_Y)
    print( "RMSEP_Mtest_pre_Y:", RMSEP_testpre_Y, "Q2_pre_Y:", Q2_testpre_Y)
    print( "RMSEP_tran_Y:", RMSEP_tran_Y, "Q2_tran_Y:", Q2_tran_Y)


    x = np.arange(7, 12)
    y = np.arange(7, 12)
    plt.figure()
    plt.plot(x, y, color='black')
    plt.plot(Ytest, Mypred, 'ro', label="M_Ypred" )
    plt.plot(Ytest, Stestypred, 'b+', label="S_Ypred")
    plt.plot(Ytest, Stestpre_ypred, 'g^', label="Trans_S_Ypred")
    plt.xlabel("Reference values")
    plt.ylabel("predicted values")
    plt.legend()

# =============================================================================
#     x = np.arange(7, 12)
#     y = np.arange(7, 12)
#     
#     plt.figure()
#     plt.plot(x, y, color='black')
#     plt.plot(Ytest, Stestypred, 'ro', label="Without transfer" )
#     plt.xlabel("Reference values",fontsize=14)
#     plt.ylabel("predicted values",fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.text(9.5,7.5,'RMSEP=%.4f'%RMSEP_S,fontsize=14) 
#     plt.legend(fontsize=14)
#       
#     plt.figure()
#     plt.plot(x, y, color='black')
#     plt.plot(Ytest, Mypred, 'ro', label="PLS" )
#     plt.xlabel("Reference values",fontsize=14)
#     plt.ylabel("predicted values",fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.text(9.5,7.5,'RMSEP=%.4f'%RMSEP_M,fontsize=14) 
#     plt.legend(fontsize=14)
#    
#     plt.figure()
#     plt.plot(x, y, color='black')
#     plt.plot(Ytest, Stestpre_ypred, 'ro', label="PCELM+PLS" )
#     plt.xlabel("Reference values",fontsize=14)
#     plt.ylabel("predicted values",fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.text(9.5,7.5,'RMSEP=%.4f'%RMSEP_Stestpre,fontsize=14) 
#     plt.legend(fontsize=14)
# 
#     plt.figure()
#     plt.plot(x, y, color='black')
#     plt.plot(Ytest, pre_test_Y, 'ro', label="PCELM+ELM" )
#     plt.xlabel("Reference values",fontsize=14)
#     plt.ylabel("predicted values",fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.text(9.5,7.5,'RMSEP=%.4f'%RMSEP_tran_Y,fontsize=14) 
#     plt.legend(fontsize=14)    
# =============================================================================




# =============================================================================
#     wavelength = np.arange(1100, 2500, 2)
#     diff = SXtestpre-MXtest
#     diff2 = SXtest-MXtest
#     plt.figure()
#     plt.subplot(211)
#     plt.plot(wavelength, diff.T)
#     plt.title('StestTEAM_Mtest')
#     plt.axis([1100, 2500, -0.08, 0.08])
#     plt.subplot(212)
#     plt.plot(wavelength, diff2.T)
#     plt.title('Stest-Mtest')
#     plt.axis([1100, 2500, -0.08, 0.08])
#     plt.show()
# =============================================================================
