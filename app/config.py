def set_conf(app):
    app.path_to_embeddings = "./app/static/data/models/glove.6B.50d.txt"
    app.path_to_saved_model = "./app/static/data/models/actual_model.pth.tar"
    app.path_to_vocabulary = "./app/static/data/vocabulary.txt"
    app.path_to_categories = "./app/static/data/categories.txt"

    app.path_to_unlabeled_data = "./app/static/data/CAPC_tokenized.csv"
    app.path_to_manually_labeled_data = "./app/static/data/manually_labeled.csv"
    app.path_to_manually_labeled_data_testset = "./app/static/data/manually_labeled_testset.csv"
    app.path_to_marker = "./app/static/data/marker.txt"

    app.using_saved_model = True

    app.train_epoch = 32
    app.batch_size = 32
    app.hidden_dim = 80

    app.batch_size = 1
    app.process_in_batch = app.batch_size
    app.current_batch = []
    app.sorted_indexes = []
    app.current_sentence = []