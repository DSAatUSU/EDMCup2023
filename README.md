# Analysis of Student Behavior and Score Prediction in ASSISTments online learning

### Authors: Aswani Yaramala, Soheila Farokhi, and Hamid Karimi

The initial dataset files are located in the `data/` folder.


### Code Setup and Requirements
You can install all the required packages using the following command:
```
    $ pip install -r requirements.txt
```


### Code for Initial Data Exploration
Code for dataset exploration and creating the plots in the Section 3.2 of the paper, is in the `Initial_data_statistics.ipynb` notebook file.


### Code for Tutoring Request and Student Performance
Code for analysis in Section 4.1 of the paper, is in the `Tutoring_analysis.ipynb` notebook file.

### Code for CCSS Skill Mastery and Student Performance
Code for analysis in Section 4.2 of the paper, is in the `AssociationRule_skills.ipynb` notebook file.


### Code for Feature Extraction
Code for extracting hand-crafted features detailed in Section 5.1. of the paper, is in the `Feature_engineering.ipynb` notebook file. This creates the dataset in setting (I). The files are saved in `saved_files/` directory. This step is necessary for creating other dataset settings and training predictive models on the dataset.


### Creating the Graph

To create the dataset in settings (II) and (III), use the following command. This will save the edge list for the graph in `saved_files/` directory a model for the learned embeddings in `models/` directory. Also, the final train and evaluation dataset for setting (II) or (III) will be saved in the `saved_files/` directory.
```
   $ python graph_dataset.py --setting <dataset_setting>
```

`--setting` is the setting of the dataset described in the paper. The value for this parameter should be either 2 for setting (II) or 3 for setting (III).


### Hyperparameter Tuning for Predictive Models and Feature Importance

To run hyperparameter tuning for 5 predictive models on the dataset in setting (I), run the code in `Predictive_models_Feat_importance.ipynb`. This notebook also contains code for creating the feature importance plot in Section 5.5.1.

To run hyperparameter tuning for 5 predictive models on the dataset in settings (II) or (III), use the following command. This will save the models for classifiers in the `models/` directory.
```
   $ python tune_models_with_embedding.py --setting <dataset_setting>
```
`--setting` is the setting of the dataset described in the paper. The value for this parameter should be either 2 for setting (II) or 3 for setting (III).


### References 
EDM Cup 2023

```
@misc{Prihar_Heffernan_2023,
  title={EDM Cup 2023},
  url={osf.io/yrwuh},
  DOI={10.17605/OSF.IO/YRWUH},
  publisher={OSF},
  author={Prihar, Ethan and Heffernan, Neil T, III},
  year={2023},
  month={Jun}
}
```
