# Comp6721-Project-G05

## Project Description:
This project aims to optimize and evaluate machine learning techniques for predicting customer subscriptions to a term deposit in the banking industry. The "Bank Marketing" dataset is used, and preprocessing steps are applied to handle missing values, encode categorical variables, and address imbalanced classes. Three machine learning techniques, namely decision trees, semi-supervised decision trees, and deep neural networks (DNN), are explored and evaluated using various performance metrics.

The results indicate that the semi-supervised decision tree model performs better than the optimized decision tree and DNN models in predicting customer subscriptions to a term deposit. The implementation details of each technique are provided, including the hyperparameters and optimization processes involved. Cross-validation is performed to assess the generalization capability of the decision tree model.

The semi-supervised decision tree algorithm follows an iterative process that involves training on labeled data, predicting labels for unlabeled data, and updating the labeled dataset based on high-confidence predictions. The DNN model consists of two hidden layers with ReLU activation functions and employs optimization techniques such as handling imbalanced data, parameter initialization, mini-batch stochastic gradient descent, and early stopping.

### Requirements

Make sure you have the following libraries installed:

- Jupyter Notebook
- Google Colab (optional, for GPU acceleration)
- PyTorch: Deep Neural Network (DNN) library
- scikit-learn (sklearn): Used for decision tree and evaluation metrics
- skorch: Library that wraps PyTorch models in scikit-learn for cross-validation
- NumPy: Library for numerical calculations and array manipulation
- Matplotlib: Comprehensive library for creating visualizations
- Pandas: Data analysis and manipulation tool
- Requests: Module for sending HTTP requests using Python
- BytesIO: Class for manipulating binary data in memory
- Seaborn: Library for statistical data visualization
- Graphviz: Open-source graph visualization software
- GridSearchCV: Technique for finding optimal parameter values from a given set
- resample: Method for tackling class imbalance in imbalanced datasets
- Random: Module for generating or manipulating random numbers
- Itertools: Module for iterating over data structures
- Time: Module for working with time-related functions
- Resnet50: Pre-trained deep learning model architecture
- Profile: Module for profiling Python code

## Instruction on How to Train/Validate Your Model

To train and validate your model on the Bank Marketing dataset, follow the steps outlined below:

1. Set up the necessary libraries and dependencies as mentioned in the requirements section.

2. get the data by running the provided cell

   ```python
   #response = requests.get(url)

   #zip_file = zipfile.ZipFile(BytesIO(response.content))
   
   #csv_file = zip_file.open('bank-full.csv')
   
   #df = pd.read_csv(csv_file, delimiter=';')
```
3. run the section on data preprocessing
4. perform the upsampling and feature scaling, and prepare tensors and data loader
  ```python
   df_tmp = X.copy()
   df_tmp['target'] = y
   df_minority = df_tmp[df_tmp['target'] == 'yes']
   df_majority = df_tmp[df_tmp['target'] == 'no']
   # Upsample the minority class
   df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
   # Combine the upsampled minority class with the majority class
   df_upsampled = pd.concat([df_majority, df_minority_upsampled])
   X_upsampled = df_upsampled.drop('target', axis=1)
   y_upsampled = df_upsampled['target']
   # Split the data into training and test sets
   
   X_trainval, X_test, y_trainval, y_test   = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=25)
   X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
   # Normalize the numerical features
   scaler = StandardScaler()
   X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
   X_val[numerical_features] = scaler.transform(X_val[numerical_features])
   X_test[numerical_features] = scaler.transform(X_test[numerical_features])
   # Convert the data to PyTorch tensors
   X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
   X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
   X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
   # Convert the 'y' variable to numerical labels
   label_encoder = LabelEncoder()
   y_train_encoded = label_encoder.fit_transform(y_train)
   y_val_encoded = label_encoder.transform(y_val)
   y_test_encoded = label_encoder.transform(y_test)
   # Convert the labels to PyTorch tensors
   
   y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
   y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
   y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
   
   train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
   val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
   test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
   
   num_classes = len(y.unique())
   
   
   batch_size = hyperparameter_values['batch_size']
   val_loader = DataLoader(val_dataset, batch_size=batch_size)
   test_loader = DataLoader(test_dataset, batch_size=batch_size)
```
4. Create and initialize the model:
   ```python
     best_hyperparameter_values={}
   best_hyperparameter_values['hidden_size']=best_result['hidden_size'].item()
   
   best_hyperparameter_values['num_hidden_layers']=best_result['num_hidden_layers'].item()
   best_hyperparameter_values['batch_norm']=best_result['batch_norm'].item()
   best_hyperparameter_values['activation']=best_result['activation'].item()
   best_hyperparameter_values['lr']=best_result['lr'].item()
   best_hyperparameter_values['batch_size']=best_result['batch_size'].item()
   best_hyperparameter_values['dropout_rate']=best_result['dropout'].item()
   best_hyperparameter_values['loss_function']=best_result['loss function'].item()
   model = NeuralNetwork(
       input_size, hyperparameter_values['hidden_size'], num_classes,
       hyperparameter_values['num_hidden_layers'], dropout_rate=hyperparameter_values['dropout_rate'],
       activation=hyperparameter_values['activation'], batch_norm=hyperparameter_values['batch_norm']
   )
   ```
After the following steps, you can use the training loop provided in the "final model" section to train and validate the model.
## Instructions on How to Run the Pre-trained Model on the Provided Sample Test Dataset

To test the pre-trained model on the provided sample test dataset, follow the steps below:

1. Open the notebook named "Using the Pretrained Model."

2. Ensure that the sample test dataset file, "sample_test_dataset.csv," is located in the same directory as the notebook.

3. Load the pre-trained model by specifying the path to the "DNN_model.pt" file.

4. Preprocess the sample test dataset if required, following the same preprocessing steps as in the training phase.

5. Feed the preprocessed sample test dataset to the pre-trained model for inference.

6. Evaluate the model's performance on the sample test dataset and analyze the results.

## Your Source Code Package in Scikit-learn and PyTorch

For the source code packages, refer to the following repositories:

- Scikit-learn (sklearn):
- Source Code: [https://github.com/scikit-learn/scikit-learn/tree/main/sklearn](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn)

- PyTorch:
- Source Code: [https://github.com/pytorch/pytorch/tree/master/torch](https://github.com/pytorch/pytorch/tree/master/torch)

Feel free to explore the repositories for more details on the source code and related resources.

## Description on How to Obtain the Dataset from an Available Download Link

To obtain the Bank Marketing dataset, follow these steps:

1. Download the dataset from the provided download link: [https://archive.ics.uci.edu/static/public/222/bank+marketing.zip](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).

2. Extract the downloaded ZIP file to your desired location.

3. The dataset should now be available in the extracted directory for further use in your project.

## Contributors

The following contributors have worked on this project:

- Professor: Arash Azarfar ([arash.azarfar@concordia.ca](mailto:arash.azarfar@concordia.ca))
- Lead TA: Soorena Salari ([soorena.salari@mail.concordia.ca](mailto:soorena.salari@mail.concordia.ca))
- Lead TA and Lab Instructor: Denisha Thakkar ([denisha.thakkar@mail.concordia.ca](mailto:denisha.thakkar@mail.concordia.ca))
- TA and Lab Instructor: Farzad Salajegheh ([farzad.salajegheh@concordia.ca](mailto:farzad.salajegheh@concordia.ca))
- TA and Lab Instructor: Y A Joarder ([ya.joarder@concordia.ca](mailto:ya.joarder@concordia.ca))

Please feel free to reach out to them if you have any questions or need further assistance.
