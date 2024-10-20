# Prediction methods comparison for Spain's energy generation and consumption
## By William Narciso

#### Video Demo: https://youtu.be/XghKCM16Q8g

### Description
Hello, since I have a Physics background I wanted to do a project that had more of a data analysis side. With this in mind I decided to compare various prediction models to see how they use data to make their predictions and compare them to each other.
The methods that I chose where:
- Linear Regression
- Decision Tree Regressor
- Random Forest
- Uniformised Data
- Gradient Boosting
- Extreme Gradient Boosting
- Bootstrapping
- Neural Networks

I also wanted to do Support Vector Regressor, but the graphs would not work for this method. The code is still in the file, if you want to see it run uncomment it, but do not create graphs.

After the program analyses the data you can create a dashboard to show the data with interactive graphs.


### How to run the program
You will see 3 python files in the folder: import_data.py, data.py, app.py that should be used in the following order:
1. import_data.py, this file cleans the starting data by removing empty rows, unecessary columns, joining the data and creating a final .csv for the data analysis;
2. data.py, this file does the data analysis. It takes a while to run, don't worry. If you want new data for the dashboard rerun this file;
3. app.py, this file crates the dashboard server that will present the graphs created during data analysis. Dash is based on flask so a cs50 environment can be used.
After creating the dashboard you can see all the graphs created and analyse them yourself, since the graphs are interative.

##Conclusions
