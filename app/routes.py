from app import app
from .modules.loader.loaddata import LoadData
from .modules.loader.loadmodel import ModelInterface

from flask import render_template, request
from flask import Response
import json

from .config import set_conf
from .model import set_model
set_conf(app)

categories = [category.split("\t") for category in open(app.path_to_categories, "r").readlines() if
              len(category) > 2]
vocabulary = [word for word in open(app.path_to_vocabulary, "r").readlines() if len(word) >= 1]


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jsdata = json.loads(request.data)
        print(jsdata)
        if "load_data" == jsdata[0]:
            app.path_to_unlabeled_data = jsdata[1]
            app.using_labels_from_unlabeled_data = bool(jsdata[2])
            app.using_saved_model = bool(jsdata[3])
            zero_marker = bool(jsdata[4])

            jsdata = ["Path to data '{}' set. (Using labels: {}, using saved model: {}). Page will be redirect.".format(
                           app.path_to_unlabeled_data,
                           app.using_labels_from_unlabeled_data,
                           app.using_saved_model)]
        else:
            return render_template('data.html', categories=categories)

        data = {
            'redirect': '/validate',
            'msg': jsdata
        }
        js = json.dumps(data)

        resp = Response(js, status=200, mimetype='application/json')
        set_model(app, categories, vocabulary, zero_marker)
        return resp

    return render_template('data.html', categories=categories)


@app.route("/validate", methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        jsdata = json.loads(request.data)
        print(jsdata)
        if "retrain" == jsdata[0]:
            print("Training.")
            X_train, _, y_train = LoadData.load_data_and_labels(app.path_to_manually_labeled_data)
            X_test, _, y_test = LoadData.load_data_and_labels(app.path_to_manually_labeled_data_testset)
            epochs, train_loss, train_acc, test_acc = ModelInterface.train_model(app.model, app.loss_function, app.optimizer, X_train, y_train, X_test, y_test,
                                                epochs=jsdata[1] if jsdata[1] > 0 else app.train_epoch, batch_size=app.batch_size)
            jsdata = {"epochs": epochs, "train_loss": train_loss, "train_acc": train_acc, "test_acc" : test_acc}
        elif "only_test" == jsdata[0]:
            X_test, _, y_test = LoadData.load_data_and_labels(app.path_to_manually_labeled_data_testset)
            report = ModelInterface.test_model(app.model, X_test, y_test)
            print(report)
            jsdata = report
        elif "save_model" == jsdata[0]:
            app.path_to_saved_model = jsdata[1]
            ModelInterface.save_model(app.model, app.optimizer, path=app.path_to_saved_model)
            jsdata = ["Model {} saved".format(app.path_to_saved_model)]
        elif "load_model" == jsdata[0]:
            app.path_to_saved_model = jsdata[1]
            ModelInterface.load_model(app.model, app.optimizer, path=app.path_to_saved_model)
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
        if app.process_in_batch == app.batch_size:
            for _ in range(app.batch_size):
                app.current_batch.append(next(app.data_loader))

            # sort them
            # app.sorted_examples = ModelInterface.get_indexes_less_confident(model, app.current_batch)
            app.process_in_batch = 0
    except StopIteration:
        return render_template('no_data.html')



    # app.current_sentence = ModelInterface.get_sentence_based_on_model(
    #                                                 app.current_batch[app.sorted_examples[app.process_in_batch][0]],
    #                                                 app.sorted_examples[app.process_in_batch])

    app.current_sentence = app.current_batch[app.process_in_batch]
    app.process_in_batch += 1

    # print (app.current_sentence)
    return render_template('index.html', sentence_list=app.current_sentence,
                           sentence_json=json.dumps(app.current_sentence), categories=categories,
                           batch_processed=app.process_in_batch, batch_size=app.batch_size)
