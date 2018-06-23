from .modules.loader.loaddata import LoadData
from .modules.procesor.savingdata import SavingData
from .modules.loader.loadmodel import ModelInterface


def set_model(app, categories, vocabulary, zero_marker):
    app.model, app.loss_function, app.optimizer = ModelInterface.create_model(embedding_path=app.path_to_embeddings,
                                                                              vocabulary=vocabulary,
                                                                              classes=categories,
                                                                              use_saved_if_found=app.using_saved_model,
                                                                              path_to_saved_model=app.path_to_saved_model,
                                                                              hidden_dim=app.hidden_dim)
    if zero_marker:
        with open(app.path_to_marker, "w") as f:
            f.write("0")

    app.data_loader = LoadData(app.path_to_unlabeled_data, app.path_to_marker).generate_data()
    app.saving_data = SavingData(app.path_to_manually_labeled_data)
