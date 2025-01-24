# Logistic Regression Classifier Visualization App  
## Demo  
Here’s an example of the application interface:  
![Demo Screenshot](https://github.com/navinbhagat322/Logistic_Regression_Visualizer/blob/main/Photo_LRC.png)

This project is a **Streamlit-based web application** that demonstrates the working of a Logistic Regression Classifier on synthetic datasets. The app allows users to configure various hyperparameters for logistic regression, visualize the dataset, and observe the decision boundaries dynamically.  

## Features  
- **Dataset Selection**: Choose between Binary or Multiclass datasets generated using `make_blobs`.  
- **Hyperparameter Tuning**: Modify Logistic Regression parameters, including:
  - Regularization (`penalty`)
  - Regularization strength (`C`)
  - Solver type (`solver`)
  - Maximum iterations (`max_iter`)
  - Multi-class strategy (`multi_class`)
  - ElasticNet ratio (`l1_ratio`)
  - Tolerance for convergence (`tol`)
  - Class weight (`class_weight`)
  - Fit intercept, dual optimization, warm start, and more.  
- **Interactive Visualization**:  
  - Plot the dataset points with decision boundaries.  
  - Observe how changes in hyperparameters affect classification results.  
- **Accuracy Calculation**: View the classifier's performance with the current settings.

## Requirements  
To run this project, you'll need the following libraries:  
- `streamlit`  
- `numpy`  
- `matplotlib`  
- `scikit-learn`  

You can install them using:  
```bash
pip install streamlit numpy matplotlib scikit-learn
```  

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/logistic-regression-visualization.git
   cd logistic-regression-visualization
   ```  

2. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

3. Open the URL displayed in your terminal to view the app in your browser.  

## File Structure  
- `app.py`: Main Streamlit application file.  
- `README.md`: Project documentation (this file).  

## How It Works  
1. **Dataset Creation**: Synthetic datasets are generated using the `make_blobs` function from `scikit-learn`.  
2. **User Interaction**: The sidebar allows users to modify hyperparameters of the Logistic Regression model.  
3. **Visualization**: The decision boundary and dataset points are plotted using `matplotlib`.  
4. **Classifier Training**: Logistic Regression is trained on the dataset, and predictions are made.  
5. **Results**: The accuracy of the classifier and an updated decision boundary plot are displayed.  


