from __future__ import print_function, division
from builtins import range

import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta

app = Flask(__name__)


class Bandit:
  def __init__(self, name):
    self.clks = 0
    self.views = 0
    self.name = name

  def sample(self):
    if self.clks == 0:
      return 0

    return float(self.clks) / float(self.views)

  def add_click(self):
    self.clks += 1

  def add_view(self):
    self.views += 1

    if self.views % 50 == 0:
      print("%s: clks=%s, views=%s" % (self.name, self.clks, self.views))

epsilon = 0.5
banditA = Bandit('A')
banditB = Bandit('B')

@app.route('/get_ad')
def get_ad():
  if np.random.random() < epsilon:
    if np.random.random() < 0.5:
      ad = 'A'
      banditA.add_view()
    else:
      ad = 'B'
      banditB.add_view()
  else:
    if banditA.sample() > banditB.sample():
      ad = 'A'
      banditA.add_view()
    else:
      ad = 'B'
      banditB.add_view()
  return jsonify({'advertisement_id': ad})


@app.route('/click_ad', methods=['POST'])
def click_ad():
  result = 'OK'
  if request.form['advertisement_id'] == 'A':
    banditA.add_click()
  elif request.form['advertisement_id'] == 'B':
    banditB.add_click()
  else:
    result = 'Invalid Input.'

  return jsonify({'result': result})


if __name__ == '__main__':
  app.run(host='127.0.0.1', port='8888')
