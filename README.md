# Projects

Hi! 😁

I have collected some of the projects I've done over my last academic years, as I think they are a good demonstration of my abilities and technical knowledge in Machine Learning, Statistical Learning and programming skills.

Here is a legend for the emojis used:  
🏆: obtained some kind of reward or recognition  
💻: code script is included  
📕: notebook is included  
📝: report is included

Finally, **ES** denotes that the project is in Spanish, while **EN** means it is completely done in English.

Hope you enjoy them!

## Current work in progress

### Master's thesis: Attention Networks for Irregularly Sampled Time Series (**EN** 💻 📝) 21/22
- Work based on this [paper](https://arxiv.org/pdf/2101.10318.pdf) (ICLR 2021)  
- To be submitted mid-june, defended mid-july

### NLP - Classigying documents using Topic Modelling (**EN** 📕) 21/22
   - Gensim, Matplotlib
   - Completed by end of May

### VAE model implementation (**EN** 📕) 21/22
   - PyTorch, Matplotlib
   - Completed by end of May

### Mysterious Kaggle competition (**EN** 📕) 21/22
  - Goal: maximize F1 score on unknown dataset, where noise and transformations have been added to the original variables
  - Completed by end of May  


## Bachelor's thesis: Time Series Anomaly Detection (**ES** 🏆 💻 📝) 20/21
- Described best-known methodologies in time series anomaly detection with emphasis on the Matrix Profile, developing the entire mathematical construction behind it
- Applied these techniques to the KDD 2021 competition using my own novel implementation, obtaining a top 20 ranking (not official)
- Awarded with Honors (best thesis in the Mathematics and Statistics degree)


## Machine Learning

### Deep Learning
#### 1. Image classification with MLPs (**EN** 📕) 21/22
   - Implemented image classifier using MLPs for the MNIST & Fashion-MNIST datasets
   - Created class with methods for forward pass, trainloop and evaluation 
   - Visualized overfitting based on train vs validation accuracy across epochs
   - Implemented early-stopping and dropout and evaluated best performance
   - PyTorch, Matplotlib 
#### 2. CNNs for CIFAR 10 (**EN** 📕) 21/22
   - Implemented LeNet 5 network and evaluated performance in train & validation
   - Regularized network to prevent overfitting (early stopping + dropout)
   - Included batch normalization to accelerate training and compared performances
   - PyTorch, Matplotlib 


### Kernel Methods
#### 1. SVMs and Spectral Clustering (**EN** 📕) 21/22
   - SVM with randomly selected labels
   - Spectral clustering with one label per cluster
   - Simulated scenario with scarce labels. Studied impact of scarcity in the results and explored strategy to try to achieve good results
   - Implemented these models on 0 vs 1 problem and even vs odd problem and commented on final discussion
   - Extra: Used KMeans instead of Spectral Clustering, discussing for which hyperparameter values they are the closest
   - Extra: Studied behavior of training SVM with labeled samples used by Spectral Clustering
   - Sklearn, NumPy, Pandas, Matplotlib 
#### 2. Kernel methods for understanding data (**EN** 📕) 21/22
   - Evaluated performance of SVM, PCA + SVM, LDA, Kernel PCA + linear SVM, KDA on face recognition task
   - Discussed results
   - Studied impact of number of components in the reduced dimension space
   - Studied performance on noise-contaminated test set
   - Extra: KDA + SVM, LDA + SVM  
   - Sklearn, Scipy, Pandas, Matplotlib 


### Kaggle Competitions
#### 1. House Prices Kaggle competition (**EN** 🏆 📕) 21/22
  - Developed several ML models with the goal of achieving the best results on the competition
  - KNN, Decision Tree, Random Forest (RF), Extra Trees (ET), Hist Gradient Boosting (HGB)
  - Default hyperparameters, grid-search, random-search, skopt, optuna, pruning
  - Compared results and drawed conclusions (speed: ET vs RF, HPT evaluation, time vs results tradeoff, additional conclusions, best model)
  - CASH implementation: ET + HGB
  - Extra: Halving hyperparameter optimization method, XGBoost
  - Achieved best competition ranking in the class

   
     
<br />
     
### 1. Scheduling optimization of hospital's inventory to tackle the COVID-19 crisis (**EN** 📕) 21/22
  - Formulated the problem mathematically as a linear optimization model
  - Implemented model in **Pyomo** and solved for a set of generated data
  - Computed and interpreted sensitivities
  - Modified problem to impose logical and conditional constraints. Implemented and solved this problem, and interpreted results
  - Implemented relaxed problem. Compared solutions and commented on them
  - Commented on impact of size of the problem and time required to solve it, as well as impact of changing one of the parameters in the model
  - Included an **interactive version** of the data generation and model development
### 2. Startup - Finding your Career (**ES** 🏆 💻 📝) 20/21
  - Planning and development of a startup aimed at predicting the ideal career for high school students using AI algorithms based on the Big Five personality traits (OCEAN), trained with 132 real responses from university students
  - Developed a model of the complete system, in addition to different aspects of the startup such as implementability, business case, social and economic impact, future developments...
  - Project preselected at the Data Science Iberian Awards
  - Language used: R


## Statistics
### 1. Top colleges ranking in the US (**EN** 📝) 21/22
  - Developed a complete statistical study about the top 650 colleges in the US
  - Gained insights about the statistical differences and similarities between private and public universities
  - Dimension reduction techniques, clustering, supervised classification
  - Language used: R
### 2. Statistical study of real data in pregnancy (**EN** 📝) 21/22
  - Dataset containing more than 40,000 real observations with many variables involved in pregnancy, being the **weight** of the newborn the central variable in the study
  - Preprocessing, descriptive analysis, model fitting, statistical inference
  - Successfully answered many clinical questions we previously posed (distribution of variables, significant statistical differences among groups, ...)
  - Language used: R
### 3. Banking Churn (**ES** 📕) 19/20
  - Study analyzing three prediction models to predict bank’s customer loyalty, concluding which is the best model and suggesting possible decisions that should be taken based on the results
  -  Language used: Python
### 4. Factorial Analysis of reading habits and use of technologies between Arab and Nordic students (**ES** 📝) 19/20
  - Factorial study on education in students from Arab and Nordic countries, comparing the different extraction and rotation methods and obtaining decisive conclusions about the differences in reading habits and use of technologies, applying the best possible model
  - Software used: SPSS
 

## Big Data
### 1. Parallelism - K-means (**EN** 💻) 21/22
  - Implemented K-means program from scratch
  - Implemented serial version, parallel version using multiprocessing, parallel version using threads
  - Measured speedup in 500,000 lines file
  - Language used: Python
### 2. Parallelism - Protein matching (**EN** 💻) 21/22
  - Implemented program to match a pattern introduced using the keyboard against all the proteins in the generated dataset
  - Implemented serial version, parallel version using multiprocessing, parallel version using threads
  - Measured speedup in 500,000 lines file
  - Language used: Python
### 3. Plan of improvement for BiciMAD service with PySpark (**ES** 📕 📝) 20/21
  - Designed and implemented a solution to optimize the supply and demand of BiciMAD service
  - Created and studied 4 seasonal maps measuring the intensity of the demand in the center of Madrid
  - Made special emphasis on the preprocessing and cleaning of data, finding useful additional information and the clarity of the maps
