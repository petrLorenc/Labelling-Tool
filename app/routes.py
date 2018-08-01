from app import app
from .modules.loader.load_data import LoadData
from .modules.loader.model_utils import ModelUtils
from .modules.model.lstm_net_pytorch import PytorchLstmNet
from .modules.model.lstm_net_keras import KerasLstmNet

from flask import render_template, request
from flask import Response
import json

from .config import set_conf
from .model import set_model

set_conf(app)


@app.route("/", methods=['GET', 'POST'])
def index():
    categories = LoadData.load_categories(app.path_to_categories)
    if request.method == 'POST':
        jsdata = json.loads(request.data)
        print(jsdata)
        status = 200
        if "load_data" == jsdata[0]:
            app.path_to_unlabeled_data = jsdata[1]
            app.using_saved_model = bool(jsdata[2])
            zero_marker = bool(jsdata[3])

            jsdata = ["Path to data '{}' set. (Using saved model: {}). Page will be redirect.".format(
                app.path_to_unlabeled_data,
                app.using_saved_model)]
        else:
            return render_template('data.html', categories=categories)

        try:
            # app.model = KerasLstmNet(app)
            app.model = PytorchLstmNet(app)
            print("tada")
            set_model(app, zero_marker)
        except Exception as e:
            raise e

        data = {
            'redirect': '/validate',
            'msg': jsdata
        }
        js = json.dumps(data)

        resp = Response(js, status=status, mimetype='application/json')
        return resp

    return render_template('data.html', categories=categories)


@app.route("/validate", methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        jsdata = json.loads(request.data)
        print(jsdata)
        if "retrain" == jsdata[0]:
            print("Training.")
            X, y = LoadData.load_data_and_labels(jsdata[2] if jsdata[2] != "" else app.path_to_manually_labeled_data)
            epochs, train_loss, train_acc, test_acc = app.model.train(X, y, epochs=jsdata[1] if jsdata[1] > 0
                                                                                             else app.train_epoch,
                                                                      batch_size=app.batch_size)
            jsdata = {"epochs": epochs, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
        elif "only_test" == jsdata[0]:
            X_test, y_test = LoadData.load_data_and_labels(app.path_to_manually_labeled_data_testset)
            report = app.model.test(X_test, y_test)
            print(report)
            jsdata = report
        elif "save_model" == jsdata[0]:
            app.path_to_saved_model = jsdata[1]
            app.model.save_model(path=app.path_to_saved_model)
            jsdata = ["Model {} saved".format(app.path_to_saved_model)]
        elif "load_model" == jsdata[0]:
            app.path_to_saved_model = jsdata[1]
            app.model.load_model(path=app.path_to_saved_model)
            jsdata = ["Model {} loaded".format(app.path_to_saved_model)]
        else:
            app.saving_data.add_item_to_file(jsdata)

        data = {
            'redirect': '/validate',
            'msg': jsdata
        }
        js = json.dumps(data)

        resp = Response(js, status=200, mimetype='application/json')

        return resp
    try:
        app.current_sentence = next(app.data_loader)
    except StopIteration:
        return render_template('no_data.html')

    if all([False if x[1] != "0" else True for x in app.current_sentence]):
        app.current_sentence = app.model.predict(app.current_sentence)

    return render_template('index.html', sentence_list=app.current_sentence,
                           sentence_json=json.dumps(app.current_sentence), categories=app.model.categories,
                           batch_processed=app.process_in_batch, batch_size=app.batch_size)
