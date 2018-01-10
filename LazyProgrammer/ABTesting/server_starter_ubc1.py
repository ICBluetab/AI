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

  def add_click(self):
    self.clks += 1

  def add_view(self):
    self.views += 1

    if self.views % 50 == 0:
      print("%s: clks=%s, views=%s" % (self.name, self.clks, self.views))


banditA = Bandit('A')
banditB = Bandit('B')

@app.route('/get_ad')
def get_ad():

  if banditA.views == 0:
    jA = 0
  else:
    jA = float(banditA.clks) / float(banditA.views) + np.sqrt(2*np.log(banditA.views + banditB.views) / banditA.views)

  if banditB.views == 0:
    jB = 0
  else:
    jB = float(banditB.clks) / float(banditB.views) + np.sqrt(2*np.log(banditA.views + banditB.views) / banditB.views)

  print("jA ", jA, " jB ", jB)
  if jA == 0 or jB != 0 and jA > jB:
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
