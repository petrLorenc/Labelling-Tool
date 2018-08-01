from .modules.loader.load_data import LoadData
from .modules.procesor.saving_data import SavingData


def set_model(app, zero_marker):
    app.model.load()

    app.path_to_marker = app.path_to_marker + app.path_to_unlabeled_data.split("/")[-1].split(".")[0] + "_marker.txt"

    if zero_marker:
        with open(app.path_to_marker, "w") as f:
            f.write("0")

    app.data_loader = LoadData(app.path_to_unlabeled_data, app.path_to_marker).generate_data()
    app.saving_data = SavingData(app.path_to_manually_labeled_data)
