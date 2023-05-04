from funcs import *
from matplotlib.colors import ListedColormap
from Anova import Anova


"""
      there is a picture that shows the outcome of this file
      to run the code might take some times since there are many things to calculate
       

"""










def main():
  cmap_l = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF", "#DDA0DD"])
  cmap_b = ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#800080" ])


  data = np.loadtxt("Arkiv/mnistsub.csv", delimiter=",", dtype=np.float64)


  X, y = data[:,:data.shape[1]-1], data[:,data.shape[1]-1]

  pros = 0.20
  y_test = y[:round(y.shape[0]*pros),]
  y = y[round(y.shape[0]*pros):,]
  X_test = X[:round(X.shape[0]*pros),]
  X = X[round(X.shape[0]*pros):,]




  Cs = [1, 10, 100, 1000, 2000, 5000, 10000]
  gammas = [10, 5, 1, 0.5, 0.1, 0.01, 0.001]
  degrees = [1, 2, 3, 4]

  sigmas = [1,1.5, 2, 2.5, 3, 3.5]
  ds = [1,2,3,4,5]
  anovas = []

  for i in sigmas:
    for j in ds:
      anovas.append(SVC(kernel=Anova(i, j).anova).fit(X,y))
    break



  linears, clfs, polys = fitter_maker(Cs, gammas, degrees)

  print("all making done")
  print("fittin all models")
  estimators_fitter(linears, X, y)
  print("linear are fitted")
  estimators_fitter(clfs, X, y)
  print("clfs (rbf) are fitted")
  estimators_fitter(polys, X, y)
  print("polys are fitted")
  estimators_fitter(anovas,X,y)
  print("anovas are fitted")



  print(len(linears), len(clfs), len(polys), len(anovas))
  ind_lin, score_l = fitter_chooser(linears, X_test, y_test)
  ind_clf, score_clf = fitter_chooser(clfs, X_test, y_test)
  ind_poly, score_poly = fitter_chooser(polys, X_test, y_test)

  ind_anova, score_anova = fitter_chooser(anovas,X_test,y_test)


  print(ind_lin, ind_clf, ind_poly, ind_anova)




  grid_l = dec_boundry(100,linears[ind_lin], X)
  grid_clf = dec_boundry(100, clfs[ind_clf], X)
  grid_poly = dec_boundry(100, polys[ind_poly], X)
  grid_anova = dec_boundry(100,anovas[ind_anova], X)

  params_l = linears[ind_lin].get_params()
  params_clf = clfs[ind_clf].get_params()
  params_poly = polys[ind_poly].get_params()
  params_anova = anovas[ind_anova].get_params()["kernel"](1,obj=True).get_params()

  title_l = f"Linear\nC = {params_l['C']}\nscore = {str(score_l)[:5]}"
  title_clf = f"rbf\nC = {params_clf['C']}\ngamma = {params_clf['gamma']}\nscore = {str(score_clf)[:5]}"
  title_poly = f"Poly\nC = {params_poly['C']}\ndegree = {params_poly['degree']}\nscore = {str(score_poly)[:5]}"
  title_anova = f"Anova\nsigma = {params_anova['sigma']}\nd = {params_anova['d']}\nscore = {str(score_anova)[:5]}"

  plt.suptitle(f"Total tests is: {len(linears) + len(clfs) + len(polys) +len(anovas)}\n"
               f"linear = {len(linears)}\nrbf = {len(clfs)}\npoly = {len(polys)}\nAnova = {len(anovas)}")

  plt.subplot(1,4,1)
  plt.title(title_l)
  points_and_grid_plot(grid_l, cmap_b, cmap_l, X, y)

  plt.subplot(1,4,2)
  plt.title(title_clf)
  points_and_grid_plot(grid_clf, cmap_b, cmap_l, X, y)

  plt.subplot(1,4,3)
  plt.title(title_poly)
  points_and_grid_plot(grid_poly, cmap_b, cmap_l, X, y)

  plt.subplot(1,4,4)

  plt.title(title_anova)
  points_and_grid_plot(grid_anova, cmap_b, cmap_l, X, y)
  plt.show()



main()

