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

After the program analyses the data you can create a dashboard to show the data with interactive graphs. I decided to use dash as it allows the code to be much more condensed, with the ability to create a comlpete dashboard with a single file. Since dash in based on Flask a CS50 envirnoment can be used to create the server.


### How to run the program
You will see 3 python files in the folder: import_data.py, data.py, app.py that should be used in the following order:
1. import_data.py, this file cleans the starting data by removing empty rows, unecessary columns, joining the data and creating a final .csv for the data analysis;
2. data.py, this file does the data analysis. It takes a while to run, don't worry. If you want new data for the dashboard rerun this file;
3. app.py, this file crates the dashboard server that will present the graphs created during data analysis. Dash is based on flask so a cs50 environment can be used.
After creating the dashboard you can see all the graphs created and analyse them yourself, since the graphs are interative.

##Conclusions
A large amount of the error can be attributed to daylight savings time, as the energy loads and generations see a 1 hour shift every time the daylight savings are changed.
The best methods for predicting energy generation and load were random forrest and uniformised data
The best methods for predicting price were extreme gradient boosting and uniformised data
If only 1 method must be selected for everything I recommend uniformised data, as it was the second best for every parameter.

###Generation
The generation could be more easily predicted hourly with the use of generation-1, as weather patterns do not usually have big variations in sucessive hours.
Generation was more dependant on the weather than the load, to be expected since Spain does have a large amount of Solar and Wind energy generation that is very dependant on weather.
Generation was also less connected to price than load. The hour seems to be correlated equally with generation and load, since the time of day for greater generation corresponds to the time of day with greater load.

###Load
The energy load was not very dependant on weather, showing that energy needs are not influenced by current weather and are much more constant.
Load was most dependant on load-1 and on hour, as expected, as it follows a pattern similar to a Gauss distribution throughout the day.
the need for energy will be higher during the day than at night time. Load has to be correlated with generation and generation-1 because when the need for energy goes up (load), so must the generation of sources that can be controlled to compensate for the increased energy needs.
Load had some relation to price as energy can be more expensive at night when load is lower and cheaper in the mornings when load ramps up.

###Price
Analysing price we can see that is was much more dependant on the total load and not generation. It did have some dependency from load-1 and almost none from generation-1.
The day of the week does not affect price very much. There is a relation between price and the hour, that can be atributed to load varying throughout the day and being closelly realted to the hour.
