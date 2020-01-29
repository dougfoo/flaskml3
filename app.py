from flask import Flask, request, jsonify
from sklearn.externals import joblib
from flask_cors import CORS
import numpy as np

#
# XGB2, ISO, LR3, RF are all from sklearn and have common input formats (18 features onehot)
# LBGM has 4 features not onehot, [carat, color, cut, clarity]
#

app = Flask(__name__)
CORS(app)

models = {}
models['XGB2'] = joblib.load('models/sklearn_diamond_xgb_model.pkl')
models['ISO'] = joblib.load('models/sklearn_diamond_iso_model.pkl')
models['LR3'] = joblib.load('models/sklearn_diamond_regr_model.pkl')
models['RF'] = joblib.load('models/sklearn_diamond_rforest_model.pkl')
# models['LBGM'] = joblib.load('models/az_automodel2.pkl')
print('loaded models', models)


@app.route('/', methods=['GET'])
def base():
    return '<div>Welcome to the Flask ML Runner -- paths available:  /models/<modelName> where modelName is one of the registered models:<P/><P/><PRE> ' +str(models)+'</PRE></div>'


# ML diamond predict models
@app.route('/models/<model>', methods=['POST'])
def predict(model):
    if (models.get(model) is None):
        print('model not found: ',model)
        return jsonify("[-1]")

    j_data = np.array(request.get_json()['data'])
    y_hat = np.array2string(models[model].predict(j_data))
    print('input: ',j_data, ', results:', y_hat)
    return y_hat


if __name__ == '__main__':
    app.run(debug=True)
