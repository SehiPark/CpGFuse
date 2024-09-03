# CpGFuse: A Holistic Approach for Accurate Identification of Methylation States of DNA CpG Sites



Anomalous DNA methylation has wide-ranging implications, spanning from neurological disorders to cancer and cardiovascular complications. Current methods for single-cell DNA methylation analysis face limitations in coverage, leading to information loss and hampering our understanding of disease associations. This study addresses the challenge of precise identification of CpG site methylation states by introduction of CpGFuse, a novel methodology. CpGFuse tackles this issue by combining information from diverse genomic features. Leveraging two benchmark datasets, we employed a careful preprocessing approach and conducted a comprehensive ablation study to assess the individual and collective contributions of DNA sequence, intercellular, and intracellular features. Our proposed model, CpGFuse, employs a Convolutional Neural Network (CNN) with an attention mechanism, surpassing existing models across HCCs and HepG2 datasets. The results highlight the effectiveness of our approach in enhancing accuracy and providing a robust tool for CpG site prediction in genomics. CpGFuse’s success underscores the importance of integrating multiple genomic features for accurate identification of methylation states of CpG site


# Raw data

The raw data used for preparing the train/validate/test datasets for HCC and HepG2 cells can be downloaded from Gene Expression Omnibus: [GSE6536](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65364)

# Methylation Data Processing Pipeline

This repository contains a comprehensive pipeline for processing and analyzing single-cell DNA methylation data from HepG2/HCC cells. The pipeline includes steps for data preprocessing, feature extraction, and dataset creation, as well as a neural network model for CpG site methylation prediction.



## Dependencies

The following esstential Python packages are required to run the pipeline:

- pandas==1.1.5
- numpy==1.19.5
- keras==2.6.0
- tensorflow==2.6.0
- biopython==1.79
- scikit-learn==0.24.1
- scipy==1.5.4

## Pipeline Steps

The full code of each step in the pipeline is available in the files [CpGFuse_HCCs.ipynb](https://github.com/SehiPark/CpGFuse/blob/main/CpGFuse_HCCs.ipynb) for HCC cells and [CpGFuse_HepG2.ipynb](https://github.com/SehiPark/CpGFuse/blob/main/CpGFuse_HepG2.ipynb) for HepG2 cells.
### 1. Data Import

We start by importing the raw methylation data files and the necessary Python libraries.

```python
import pandas as pd
import pickle
import numpy as np
import Genome
from concurrent.futures import ThreadPoolExecutor

file_paths = [
    'methylation_data/GSM2039756_scTrio_HepG2_1_RRBS.single.CpG.txt',
    'methylation_data/GSM2039758_scTrio_HepG2_2_RRBS.single.CpG.txt',
    'methylation_data/GSM2039760_scTrio_HepG2_3_RRBS.single.CpG.txt',
    'methylation_data/GSM2039762_scTrio_HepG2_4_RRBS.single.CpG.txt',
    'methylation_data/GSM2039764_scTrio_HepG2_5_RRBS.single.CpG.txt',
    'methylation_data/GSM2039766_scTrio_HepG2_6_RRBS.single.CpG.txt'
]
```

### 2. Data Processing
The processing pipeline filters data based on read counts, retaining only data points with four or more reads. It then selects the necessary columns (chromosome, position, strand, and label) and binarizes the labels.
```python
def process_and_save(file_paths, chro, position, strand, read, label):
    for i, file_path in enumerate(file_paths):
        data_frame = pd.read_csv(file_path, sep='\t', header=None)
        processed_data_frame = data_frame[data_frame[read] >= 4]
        selected_columns = processed_data_frame[[chro, position, strand, label]]
        selected_columns.columns = ['chro', 'position', 'strand', 'label']
        selected_columns['label'] = selected_columns['label'].apply(lambda x: 1 if x >= 0.5 else 0)
        output_file_path = f'picked_columns_HepG2_cell{i+1}.csv'
        selected_columns.to_csv(output_file_path, sep='\t', index=False, header=False)

process_and_save(file_paths, chro, position, strand, read, label)

```
Processed data is saved in CSV format for further analysis.




### 3. Chromosome and Cell Segmentation

The processed data is segmented by chromosomes and cells. This segmentation helps in organizing data and facilitates efficient processing in subsequent steps.

```python
def split(files, chromosomes, chro_col, position_col, label_col):
    # ...

```
Positions and labels are saved as dictionary objects and stored in pickle files.



### 4. Position and Range Calculation

Unique positions are combined across all cells per chromosome, and the minimum and maximum positions are calculated for each chromosome to define the ranges covered by the data.


```python
def positions(number_of_cells, chromosomes, position_col):
    # ...

```
The Imin and Imax values for each chromosome are stored in pickle files for later use.





### 4. DNA Features

One-hot encoding is applied to the DNA sequences surrounding each CpG site.



```python
def onehot(seq):
    # ...

```
The encoded sequences are saved as pickle files.





### 5. CpG Features

Intercellular and intracellular features are extracted based on the CpG sites' positions and methylation states.

```python
def cpg_augment(chromosomes, number_of_cells, position_col, label_col):
    # ...

```
Features are saved in NumPy and pickle formats.


### 6. Dataset Creation


The pipeline prepares training, test, and validation datasets by combining features from all chromosomes and cells.


```python
def dataset(number_of_cells, training_set, test_set, validation_set, dna_size, intra_size):
    # ...

```
The datasets are saved as pickle files, ready for model training.




### 7. Model Definition


The core of this project is a convolutional neural network (CNN) designed to predict the methylation state of CpG sites. The model integrates DNA sequence features, intercellular features, and intracellular features.



```python
def CpGFuse_model():
    # ...

```




### 8. Training, Validation, and Testing

The prepared datasets are used to train, validate, and test the CNN model. Keras is employed for model building and training, with checkpoints and early stopping to optimize performance.

```python
import pickle
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve


def train_and_evaluation(number_of_cells):
    for i in range(number_of_cells):
        with open('training_set_HepG2_cell'+str(i+1)+'.pkl', 'rb') as f:
            train=pickle.load(f)
            x1_train=train[0]
            x2_train=train[1]
            x3_train=train[2]
            y_train=train[3]
        with open('test_set_HepG2_cell'+str(i+1)+'.pkl', 'rb') as f:
            test=pickle.load(f)
            x1_test=test[0]
            x2_test=test[1]
            x3_test=test[2]
            y_test=test[3]
        with open('val_set_HepG2_cell'+str(i+1)+'.pkl', 'rb') as f:
            val=pickle.load(f)
            x1_val=val[0]
            x2_val=val[1]
            x3_val=val[2]
            y_val=val[3]
            
        # Model Initialization    
        model = CpGFuse_model()
        
        filename='best_cpgfuse_HepG2_cell'+str(i+1)+'.h5'
        checkpoint=ModelCheckpoint(filename,
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True,
                                   mode='min')
        earlystopping= EarlyStopping(monitor='val_loss',
                                     patience=10)

        op= tf.keras.optimizers.SGD(learning_rate= 0.05) #RMS, prop, SGD, 
        model.compile(loss='binary_crossentropy',optimizer= op, metrics=['accuracy'])

        history = model.fit([x1_train,x2_train, x3_train],y_train, batch_size=32, epochs=80, validation_data =([x1_val, x2_val, x3_val], y_val), callbacks=[checkpoint, earlystopping])
        model.load_weights(filename)
        
        loss, accuracy = model.evaluate([x1_test, x2_test, x3_test], y_test)
        
        print(f'Test Loss for cell {i + 1}: {loss}')
        print(f'Test Accuracy for cell {i + 1}: {accuracy}')
        
        y_pred = model.predict([x1_test, x2_test, x3_test])
        y_pred_edit = np.where(y_pred >= 0.5, 1, 0)
        y_pred_edit = y_pred_edit.reshape(-1,)
        cm = confusion_matrix(y_test, y_pred_edit)
        print('Confusion Matrix : \n', cm)
        # [0,0]: true negative
        # [0,1]: false positive
        # [1,0]: false negative
        # [1,1]: true positive
        
        total=sum(sum(cm))
        
        #####from confusion matrix calculate accuracy
        accuracy=(cm[0,0]+cm[1,1])/total
        print ('Accuracy_cell'+str(i+1)+' : ', accuracy)
        sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
        print('Sensitivity_cell'+str(i+1)+' : ', sensitivity )
        specificity = cm[0,0]/(cm[0,1]+cm[0,0])
        print('Specificity_cell'+str(i+1)+' : ', specificity)
        precision = cm[1,1]/(cm[1,1]+cm[0,1])
        print('Precision_cell'+str(i+1)+' : ', precision)
        print('MCC_cell'+str(i+1)+' : ', matthews_corrcoef(y_test, y_pred_edit))
        print('f1_score_cell'+str(i+1)+' : ', f1_score(y_test,y_pred_edit))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        AUROC=metrics.auc(fpr, tpr)
        print('ROC AUC_cell'+str(i+1)+' : ', AUROC)
        pr_auc = average_precision_score(y_test, y_pred)
        print('PR AUC_cell'+str(i+1)+' : ', pr_auc)

```


## The trained models

The trained models can be downloaded from [trained_models](https://github.com/SehiPark/CpGFuse/tree/main/trained_models).
- [HCCs trained models](https://github.com/SehiPark/CpGFuse/tree/main/trained_models/HCC_models)
- [HepG2 trained models](https://github.com/SehiPark/CpGFuse/tree/main/trained_models/HepG2_models)



## Testset predictions

The trained models can be downloaded from [trained_models](https://github.com/SehiPark/CpGFuse/tree/main/trained_models).
- [HCCs](https://github.com/SehiPark/CpGFuse/blob/main/preidction_HCCs.zip)
- [HepG2](https://github.com/SehiPark/CpGFuse/blob/main/prediction_hepg2.zip)




