# Labelling Tool with a little bit of Active Learning
Web based labelling tool with active learning. Model is defined in [LSTMNet.py](app/modules/model/lstm_net.py)

The trained model is trying to predict labels for your examples and ask for human labelling for these which he is most uncertain. Loading batch size is set to 20.

Data for labelling are not included because of licence. 
The tool is still in development but can be used to speed up labelling data. 

[Design 1.0](design01.pdf)

## Usage

If you want to run in on your machine, download this repository and run:

    export FLASK_APP=application.py; export FLASK_DEBUG=1; python -m flask run
    
Then you can access website at your localhost address. For example [http://127.0.0.1:5000/ ](http://127.0.0.1:5000/)

For own model see [abstract_model.py USAGE](app/modules/model/base/USAGE.md)

  * Pytorch implementation: [lstm_net_pytorch](app/modules/model/lstm_net_pytorch.py)
    * Legacy code, slow
  * Keras implementation: [lstm_net_keras](app/modules/model/lstm_net_keras.py)
    * Improved loading and creating model, more faster
    
### Input format of data

    what	0	0
    are	0	0
    you	1	B-pronoun
    talking	0	0
    about	1	B-generic_entity
    .	0	0
    
    let's	0	0
    talk	0	0
    about	0	0
    politics	1	B-topic
    .	0	0

Between each example is newline and each word is in this format:

    what\t0\t0\n
    
## Workflow

After labelling some data you can retrain your model (it does not save a model). Then you can test the model and choose to save new one on load the previous one. After saving there is no backup of old model (so by careful).

## Todos

  * Hard-written OOV vocabulary (Repair in Keras model)