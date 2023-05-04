import numpy as np


class fun:
  reg = None

  def __init__(self, reg):
    self.reg = reg

  def __optimize(self ,a, diff):
    beta_t = np.zeros(self.reg.Xen.shape[1])
    iter = 0
    gg = [beta_t]
    while True:
      gg.append(self.reg.gradient_docent(gg[iter], a))
      iter+=1
      if abs(self.reg.cost_fun(gg[iter-1]) - self.reg.cost_fun(gg[iter])) < diff:
        print("enough optimized")
        break
      if self.reg.cost_fun(gg[iter]) > self.reg.cost_fun(gg[iter-1]):
        print("came there we want the minimum value")
        break
      if iter == 10*10e6:
        print("iterations", iter)
        break
      print(self.reg.cost_fun(gg[iter]), self.reg.cost_fun(gg[iter-1]))

    return iter, gg[iter-1], gg



  def find(self,start_alpha = 0.01, learning_rate = 0.01, diff = 1*10e-9, max_iter = 1000):
    beta = 0
    beta_start = np.zeros(self.reg.Xen.shape[1])
    a = start_alpha
    iter = 0
    g = [beta_start]
    betas = [beta_start]
    while True:
      g.append(self.reg.gradient_docent(g[iter], a))
      iter+=1

      if iter > max_iter:
        print(a, self.reg.cost_fun(g[-1]))
        a+=learning_rate
        iter = 0
        betas.append(g[-1])
        g = [beta_start]
      elif abs(self.reg.cost_fun(g[iter-1]) - self.reg.cost_fun(g[iter])) < diff: ## has been enough stabele now for this alfa
        beta = g[-1]
        break

      elif self.reg.cost_fun(g[iter]) > self.reg.cost_fun(g[iter-1]):
        if iter == 10:
          print("the start alpha is big")
          print("alpha is now 0.01")
          print("calculating the value of alpha and iterations")
          a = 0.01
          g = [beta_start]
          iter = 0
        else:
          if self.reg.cost_fun(betas[-1]) < self.reg.cost_fun(g[-2]):
            a -= learning_rate # the perv step was better
            iter, beta ,g = self.__optimize(a, diff) # since it did not break before we did not get
            # to the best value and it could be better if iter was more than 1000
            break


    return  a, iter, beta, g
