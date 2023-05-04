import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from A2 import NON_LOG_REG
from A_N_func import fun



cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


data = np.loadtxt("A2_datasets_2022/microchips.csv", delimiter=",", dtype=np.float64)

X_0, y = data[:,:2], data[:,2]

############
#  Task 1  #
############
plt.figure()
plt.scatter(X_0[:,0], X_0[:,1], c=y)

############
#  Task 2  #
############
x01 = X_0[:,0] **2
x02 = X_0[:,1] **2
xx = X_0[:,0] * X_0[:,1]

X_1 = np.column_stack((X_0, x01,xx,x02))

###################

X,y = data[:,:2], data[:,2]
reg = NON_LOG_REG(X_1,y)


reg.degree = 2 ## to avoid problem in decison boundry since it use mapFeature that need a degree
# alpha = 7.56, Niter = 1008
# how I find this params is shown in the file A_N_func

              ######################
            #### Gradient descent ####
#################################################
start_beta = np.zeros(X_1.shape[1] + 1)         #
a = 7.56                                        #
iter = 1008                                     #
g = [start_beta]                                #
for i in range(iter):                           #
  g.append(reg.gradient_docent(g[i],a))         #
#################################################

reg.beta = g[-1] ## assiging beta of gradient descent to the obj

print("degree 2")
print("alpha value:", a, "|| iterations: ",iter)

print("cost", reg.cost_fun(reg.beta))

          ############################
          # Cost over iteration plot #
###################################################
plt.figure()                                      #
plt.suptitle("degree 5")

plt.subplot(1,2,1)                                #
                                                  #
yt = np.arange(0,iter+1, 1).tolist()              #
for c in range(len(g)):                           #
  g[c] = reg.cost_fun(g[c])                       #
                                                  #
plt.plot( yt,g)                                   #
                                                  #
plt.subplot(1,2,2)                                #
###################################################



                              ####################
                            ## find train errors ##
###############################################################################
er = 0                                                                        #
for i in range(reg.Xen.shape[0]):                                             #
  if (round(reg.sigmoid(reg.Xen[i].dot(reg.beta)))) != int(y[i]): er+=1       #
                                                                              #
print("train errors:", er)                                                    #
###############################################################################




                            #######################
                          ## Decsison boundry plot ##
###################################################################################
plt.title(f"train errors: {er}")                                                  #
grid = reg.dec_boundry(100,reg.beta)                                              #
                                                                                  #
                                                                                  #
plt.scatter(reg.Xn[:, 0], reg.Xn[:,1], c = y, marker=".", cmap=cmap_bold)         #
plt.imshow(grid.T,origin='lower',                                                 #
                   extent=(min(reg.Xen[:, 1])- 0.5, max(reg.Xen[:, 1]) + 0.5,     #
                           min(reg.Xen[:, 2])- 0.5, max(reg.Xen[:, 2])+ 0.5),     #
           cmap=cmap_light                                                        #
           )                                                                      #
# plt.show()                                                                       #
###################################################################################


########################################################################################################################

############
#  Task 3  #
############
# the function is found in the class NON_LOG_REG in the fila A2
# I just changed it little to be good for my model.




############
#  Task 4  #
############

reg = None
reg = NON_LOG_REG(X_0, y, 5)

# ff = fun(reg)
# a, i,beta,g = ff.find(0.01) ## finds the best alpha and iterations

# print(a,i)
# exit()

print("#"*20)
print("degree 5")


              ######################
            #### Gradient descent ####
#################################################
start_beta = np.zeros(reg.Xen.shape[1])         #
a = 7.34                                        #
iter = 90382                                    #
g = [start_beta]                                #
for i in range(iter):                           #
  g.append(reg.gradient_docent(g[i],a))         #
#################################################

reg.beta = g[-1] ## assiging beta of gradient descent to the obj

print("alpha value:", a, "|| iterations: ",iter)

print("cost", reg.cost_fun(reg.beta))


          ############################
          # Cost over iteration plot #
###################################################
plt.figure()                                      #
plt.suptitle("degree 5")
plt.subplot(1,2,1)                                #
                                                  #
yt = np.arange(0,iter+1, 1).tolist()              #
for c in range(len(g)):                           #
  g[c] = reg.cost_fun(g[c])                       #
                                                  #
plt.plot( yt,g)                                   #
                                                  #
plt.subplot(1,2,2)                                #
###################################################




                              ####################
                            ## find train errors ##
###############################################################################
er = 0                                                                        #
for i in range(reg.Xen.shape[0]):                                             #
  if (round(reg.sigmoid(reg.Xen[i].dot(reg.beta)))) != int(y[i]): er+=1       #
                                                                              #
print("train errors:", er)                                                    #
###############################################################################


                            #######################
                          ## Decsison boundry plot ##
###################################################################################
plt.title(f"train errors: {er}")                                                  #
grid = reg.dec_boundry(100,reg.beta)                                              #
                                                                                  #
                                                                                  #
plt.scatter(reg.Xn[:, 0], reg.Xn[:,1], c = y, marker=".", cmap=cmap_bold)         #
plt.imshow(grid.T,origin='lower',                                                 #
                   extent=(min(reg.Xen[:, 1])- 0.5, max(reg.Xen[:, 1]) + 0.5,     #
                           min(reg.Xen[:, 2])- 0.5, max(reg.Xen[:, 2])+ 0.5),     #
           cmap=cmap_light                                                        #
           )                                                                      #
plt.show()                                                                       #
###################################################################################
