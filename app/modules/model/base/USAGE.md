see [abstract_model.py](./abstract_model.py)

## Set model, loss function and optimizer

load()

    classes: [['movie', 'Movies\n'], ['actor', 'Movies\n'], ...]
    vocabulary: ['one', 'two', 'three' ...]
    zero_marker: boolean

->

    None

-----------------------
-----------------------

## Label new data

predict()

    X: [['no', '0', '0'], ['i', '0', '0'], ["didn't", '0', '0'], ['END', '0', '0']]

->

    return: [['no', '1', 'B-book'], ['i', '1', 'I-character'], ["didn't", '1', 'I-genre'], ['END', '1', 'I-genre']]

-----------------------
-----------------------

## Test model

test()

    X_test:		[list(['yes', 'i', 'do']) list(['no', 'thank', 'you'])
                 list(['no', 'thank', 'you']) list(['yes', 'i', 'do']),...]
    
    y_test:		[list(['0', '0', '0']) list(['0', '0', 'B-pronoun'])
			 list(['0', '0', 'B-pronoun']) list(['0', '0', '0']),...]


->

    import sklearn.metrics
    raw_report = sklearn.metrics.classification_report(self.model.prepare_targets(y_test, batch=True), tag_scores)
    
    report_data = []
    lines = raw_report.split('\n')
    for line in lines[2:-3]:
        row = {}
        print (line)
        row_data = line.split('      ')
        if len(row_data) > 0:
            row['class'] = self.model.return_class_from_target([int(row_data[1])])[0]
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])
            report_data.append(row)
    
    return report_data

-----------------------
-----------------------

## Train model

train()

    X_train:	[list(['do', 'you', 'have', 'a', 'boyfriend'])
                 list(["that's", 'a', 'good', 'one']) list(['yes', 'tell', 'me', 'more'])...]
    
    y_train:	[list(['0', '0', '0', '0', 'B-generic_entity']) list(['0', '0', '0', '0'])
                 list(['0', '0', '0', '0']) list(['0', '0', 'B-generic_entity'])...]
    
    X_test:		[list(['yes', 'i', 'do']) list(['no', 'thank', 'you'])
                 list(['no', 'thank', 'you']) list(['yes', 'i', 'do']),...]
    
    y_test:		[list(['0', '0', '0']) list(['0', '0', 'B-pronoun'])
                 list(['0', '0', 'B-pronoun']) list(['0', '0', '0']),...]
    
    epochs=100
    batch_size=32

->


    epochs: 	<class 'list'>: [0, 1, ...]
    train_loss: <class 'list'>: [0.09919151492502498, 0.04627711197425579, ...]
    train_acc:	<class 'list'>: [70.45454545454545, 80.63241106719367, ...]
    test_acc:	<class 'list'>: [80.67375886524822, 80.67375886524822, ...]


