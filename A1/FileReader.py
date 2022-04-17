import numpy as np


class File_Reader():
  """_summary_ Opens the file and reads the data

  Returns:
      numpy array: return numpy array of the data in file
  """
  data = None
  file_path = None

  def __init__(self, file_path):
    self.file_path = file_path
    self.load_data()

  def load_data(self):
    self.data = np.loadtxt(self.file_path, dtype=np.float64, delimiter=",")

  def get_data(self):
    return self.data

  def get_OK_set(self):
    d = []

    for row in self.get_data():
      if row[-1] == 1:
        d.append([row[0], row[1]])
    return np.array(d)

  def get_fail_set(self):
    d = []

    for row in self.get_data():
      if row[-1] == 0:
        d.append([row[0], row[1]])
    return np.array(d)
