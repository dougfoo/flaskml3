from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print('***** test runner for nlp_mod -- in prod use app.py *****')

import nlp_mod

if __name__ == '__main__':
    app.run(debug=True)