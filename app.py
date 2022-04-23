from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import torch
from base64 import b64encode
import os
from clf1 import predict
from model import LSTMRating
from generate_html import generate_html, get_count_html
import torch.nn as nn

import __main__

__main__.LSTMRating = LSTMRating


app = Flask(__name__)
model = torch.load('fullmodel.pt',map_location = torch.device("cpu"))
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        userId = request.form['search']
        # if search button hit, call the function get_image_class
        one, two, three, four, five = predict(userId, model)

        return render_template('reclist.html', one=one, two=two, three=three, four=four, five=five)
    return render_template('home.html')

if __name__ == '__main__' :
    app.run(host='127.0.0.1', port=8080, debug=True)