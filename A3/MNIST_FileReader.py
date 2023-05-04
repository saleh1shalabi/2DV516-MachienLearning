
# for the extraction of the MNIST files I used some code
# of the source bellow on GitHub
# https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390


import os
from PIL import Image as img
import imageio
import csv
import gzip
import numpy as np


def extract_data(filename, img_num):

  with gzip.open(filename) as file:
    file.read(16)
    buf = file.read(28 * 28 * img_num)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(img_num, 28, 28, 1)
    return data


def extract_labels(filename, img_num):

  with gzip.open(filename) as file:
    file.read(8)
    buf = file.read(1 * img_num)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def extract():
  train_data = extract_data("train-images-idx3-ubyte.gz", 60000)
  train_labels = extract_labels("train-labels-idx1-ubyte.gz", 60000)
  test_data = extract_data("t10k-images-idx3-ubyte.gz", 10000)
  test_labels = extract_labels("t10k-labels-idx1-ubyte.gz", 10000)

  if not os.path.isdir("mnist/train-images"):
    os.makedirs("mnist/train-images")
    with open("mnist/train-labels.csv", 'w') as file:
      writer = csv.writer(file, delimiter=',', quotechar='"')
      for i in range(len(train_data)):
        imageio.imwrite("mnist/train-images/" + str(i) + ".jpg", train_data[i][:, :, 0])
        writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])

  if not os.path.isdir("mnist/test-images"):
    os.makedirs("mnist/test-images")
    with open("mnist/test-labels.csv", 'w') as file:
      writer = csv.writer(file, delimiter=',', quotechar='"')
      for i in range(len(test_data)):
        imageio.imwrite("mnist/test-images/" + str(i) + ".jpg", test_data[i][:, :, 0])
        writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])


def get_train_set(samples):
  extract()
  data = []

  labels = []

  with open("mnist/train-labels.csv", "r") as file:
    sample = 0
    row = file.readline()
    while row != None:
      if sample == samples:
        break
      if row == "":
        break
      row = row.split(",")
      labels.append(int(row[1]))
      i = img.open("mnist/" + row[0])
      pic = np.asarray(i).reshape(28*28)
      data.append(pic)
      row = file.readline()
      sample += 1

  data = np.array(data)
  labels = np.array(labels)
  data = np.column_stack((data, labels))
  return data



def get_test_data(samples):

  data = []
  labels = []

  with open("mnist/test-labels.csv", "r") as file:
    sample = 0
    row = file.readline()
    while row != None:
      if sample == samples:
        break
      if row == "":
        break
      row = row.split(",")
      labels.append(int(row[1]))
      i = img.open("mnist/" + row[0])
      pic = np.asarray(i).reshape(28*28)
      data.append(pic)
      row = file.readline()
      sample += 1
  data = np.array(data)
  labels = np.array(labels)
  data = np.column_stack((data, labels))

  return data


