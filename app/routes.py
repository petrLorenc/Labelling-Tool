from app import app
from .modules.loader.loaddata import LoadData
from .modules.procesor.savingdata import SavingData
from .modules.loader.loadmodel import ModelInterface

from flask import render_template, request, redirect, g
from flask import Response, stream_with_context
import json

app.path_to_embeddings = "./app/static/data/models/glove.6B.50d.txt"
app.path_to_saved_model = "./app/static/data/models/actual_model.pth.tar"
app.path_to_vocabulary = "./app/static/data/vocabulary.txt"
app.path_to_categories = "./app/static/data/categories.txt"

app.path_to_unlabeled_data = "./app/static/data/CAPC_tokenized.csv"
app.path_to_manually_labeled_data = "./app/static/data/manually_labeled.csv"
app.path_to_manually_labeled_data_testset = "./app/static/data/manually_labeled_testset.csv"
app.path_to_marker = "./app/static/data/marker.txt"

app.train_epoch = 25
app.batch_size = 32
app.hidden_dim = 80

categories = [category.split("\t") for category in open(app.path_to_categories, "r").readlines() if
              len(category) > 2]
vocabulary = [word for word in open(app.path_to_vocabulary, "r").readlines() if len(word) >= 1]

model, loss_function, optimizer = ModelInterface.create_model(embedding_path=app.path_to_embeddings,
                                                              vocabulary=vocabulary,
                                                              classes=categories,
                                                              use_saved_if_found=True,
                                                              path_to_saved_model=app.path_to_saved_model,
                                                              hidden_dim=app.hidden_dim)

data_loader = LoadData(app.path_to_unlabeled_data, app.path_to_marker).generate_data()

batch_size = 20
app.process_in_batch = 20
app.current_batch = []
app.sorted_indexes = []
app.current_sentence = []

saving_data = SavingData(app.path_to_manually_labeled_data)


@app.route("/", methods=['GET', 'POST'])
def index():
    return redirect("/validate")


@app.route("/validate", methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        jsdata = json.loads(request.data)
        print(jsdata)
        if "retrain" == jsdata[0]:
            print("Training.")
            X_train, _, y_train = LoadData.load_data_and_labels(app.path_to_manually_labeled_data)
            losses = ModelInterface.train_model(model, loss_function, optimizer, X_train, y_train,
                                                epochs=app.train_epoch, batch_size=app.batch_size)
            jsdata = [str(losses) + " - based on " + str(len(X_train)) + " examples"]
        elif "only_test" == jsdata[0]:
            X_test, _, y_test = LoadData.load_data_and_labels(app.path_to_manually_labeled_data_testset)
            score = ModelInterface.test_model(model, X_test, y_test)
            jsdata = [score]
        elif "save_model" == jsdata[0]:
            ModelInterface.save_model(model, optimizer, path=app.path_to_saved_model)
            jsdata = ["Model saved"]
        elif "load_model" == jsdata[0]:
            ModelInterface.load_model(model, optimizer, path=app.path_to_saved_model)
            jsdata = ["Model Loaded"]
        else:
            saving_data.add_item_to_file(jsdata)

        data = {
            'redirect': '/validate',
            'msg': jsdata
        }
        js = json.dumps(data)

        resp = Response(js, status=200, mimetype='application/json')

        return resp

    if app.process_in_batch == batch_size:
        for _ in range(batch_size):
            app.current_batch.append(next(data_loader))
        # sort them
        app.sorted_examples = ModelInterface.get_indexes_less_confident(model, app.current_batch)
        app.process_in_batch = 0

    app.current_sentence = ModelInterface.get_sentence_based_on_model(
                                                    app.current_batch[app.sorted_examples[app.process_in_batch][0]],
                                                    app.sorted_examples[app.process_in_batch])
    app.process_in_batch += 1

    # print (app.current_sentence)
    return render_template('index.html', sentence_list=app.current_sentence,
                           sentence_json=json.dumps(app.current_sentence), categories=categories,
                           batch_processed=app.process_in_batch, batch_size=batch_size)
