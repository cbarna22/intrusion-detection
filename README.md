# intrusion-detection
Data mining project to identify different types of cyber attacks using network port data.

In this project, I wrote a Python script to classify different types of cyber attacks using network port data and various data mining algorithms. The project steps of data collection, data pre-processing, model creation, and testing are described below. This project uses the UNR-IDD Intrusion Detection Dataset from Kaggle: https://www.kaggle.com/datasets/tapadhirdas/unridd-intrusion-detection-dataset


1. Data Collection

In the data collection step, the dataset was first sourced from Kaggle. It was created by researchers from the University of Nevada, Reno, and consists of 34 columns and 37,411 instances of network port statistics. All of the columns are numeric except for the Switch ID and Port Number columns, which were converted from string to float. The dataset includes six different classes representing different types of cyberattacks: PortScan, TCP-SYN, Blackhole, Overflow, Diversion, and Normal, meaning no attack.

2. Data Pre-Processing

The first part of the data pre-processing was data exploration, which included plotting the count of different classes and counting the number of missing or null values. The dataset has no missing values. Prior to balancing, PortScan, TCP-SYN, and Blackhole attacks had the highest frequency at over 8000 instances each. Diversion and Normal were somewhat lower at approximately 4000-6000, and Overflow attacks were the lowest at approximately 800. 

Next, I removed outliers using quantile-based flooring and capping, which allowed me to remove any values above or below a given percentile. I chose to remove rows with values above a percentile of 99.9 or below a percentile of 0.1. 

After removing outliers, I balanced the different classes using upsampling and downsampling, resulting in a count of 4000 for each class, for a total of 24,000 instances. After balancing, I normalized the data using MinMaxScaler, which transformed the values in the dataset to a 0-1 scale. This allowed the models to process the data more easily.

I then did feature selection. I first dropped any constant features, meaning columns which had the same value for every instance. Feature importance was then quantified using mutual information classification, and I dropped the less important features until only the twelve most important remained.

The last part of preprocessing was hyperparameter tuning. The hyperparameters for the various models were selected using a grid search. 

3. Model Creation

The third step of the project was model creation. The script uses Gaussian Naive Bayes, K-Nearest Neighbors, Decision Tree, and Random Forest models. These algorithms were chosen because they are suitable for multi-class classification. After creating each model, I fit the training data to them. For the KNN, Decision Tree, and Random Forest models I also had to ensure that I was using the results of the hyperparameter tuning when creating the models.

4. Testing

The final step in the project was testing how the models performed on the data. I used an 80-20 stratified train-test split for testing, which means that I used 80% of the data for training and 20% for testing. Since it was stratified, this ensured that an equal distribution of each class would be in both the training and testing data. 

To assess the accuracy, the script displays a classification report which included the precision, recall, and f-1 score for each model. It also displays a confusion matrix which shows the number of true positives, true negatives, false positives, and false negatives each model output for each class.

