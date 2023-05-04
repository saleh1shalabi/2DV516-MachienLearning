import numpy as np


class Anova():

  """
  the anova kernel
  """
  def __init__(self,sigma, d):
    self.sigma = sigma
    self.d = d
    self.first = 0

  def get_params(self):
    return {"sigma":self.sigma, "d":self.d}

  def anova(self, X, y=None, obj = False):
    """
      the function that computes the kernek
    """
    if self.first == 0: ## needed since the set must be saved of the fit

      self.X_train = X
      self.first+=1

    if obj: ## to be able to get back the object to get params
     return self
    else :
      gram = np.zeros([X.shape[0],self.X_train.shape[0]])
      for i in range(X.shape[0]):
        for j in range(self.X_train.shape[0]):

          gram[i, j] = np.sum(np.exp((-self.sigma*  ((X[i]- self.X_train[j]) ** 2))  )) ** self.d


    return gram

