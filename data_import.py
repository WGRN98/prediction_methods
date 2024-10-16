import pandas as pd
import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))

#Importing weather data
weather=pd.read_csv('data/weather.csv')
energy=pd.read_csv('data/energy.csv')
#print(weather.info())
#print(energy.info())
#print(weather)
#print(energy)


#Setting time to datetime and using as index, timezone data can be removed, all times are in local time
weather['time']=pd.to_datetime(weather['time'])
weather=weather.set_index('time', drop=True)
energy['time'] = energy['time'].astype(str).str[:-6]
energy['time']=pd.to_datetime(energy['time'])
energy=energy.set_index('time', drop=True)
#print(weather.info())
#print(energy.info())
#print(weather)
#print(energy)


#weather_main and weather_id are definitions of weather, does not have actual usable data
#temp_min and temp_max are the same as temp
#There is no fossil coal-derived gas, fossil oil shale, fossil peat, geothermal, marine or offshore wind generation, removing empty columns and forecasts
#Hydro pumped storage aggregated column is empty in every row
weather=weather.drop(columns=['weather_main', 'temp_min', 'temp_max', 'weather_id'])
energy=energy.drop(columns=['generation fossil coal-derived gas','generation fossil oil shale','generation fossil peat','generation geothermal','generation marine','generation wind offshore','generation hydro pumped storage aggregated', 'forecast solar day ahead', 'forecast wind offshore eday ahead', 'forecast wind onshore day ahead', 'total load forecast', 'price day ahead'])
#Total energy generation
energy['total generation']=energy['generation biomass']+energy['generation fossil brown coal/lignite']+energy['generation fossil gas']+energy['generation fossil hard coal']+energy['generation fossil oil']+energy['generation hydro pumped storage consumption']+energy['generation hydro run-of-river and poundage']+energy['generation hydro water reservoir']+energy['generation nuclear']+energy['generation other']+energy['generation other renewable']+energy['generation solar']+energy['generation waste']+energy['generation wind onshore']


#Separating weather by city
weather_madrid=weather[weather['city_name'] == 'Madrid']
weather_valencia=weather[weather['city_name'] == 'Valencia']
weather_seville=weather[weather['city_name'] == 'Seville']
weather_barcelona=weather[weather['city_name'] == 'Barcelona']
weather_bilbao=weather[weather['city_name'] == 'Bilbao']


#Grouping weather by mean for the 5 cities. considering mean values for country
weather=pd.concat([weather_madrid, weather_valencia, weather_seville, weather_barcelona, weather_bilbao]).groupby(level=0).mean(numeric_only=True)

#There is a variation of around 30 non-null rows in the energy columns, not a significant number compared to 35000 entries
energy=energy.dropna()
weather=weather.dropna()
#print(weather.info())
#print(energy.info())
#print(weather)
#print(energy)


#Merging the data
data=energy.merge(weather, left_index=True, right_index=True, how='inner')

#print(weather.info())
#print(energy.info())
#print(data.info())

#Exporting csv for each dataframe
weather.to_csv('data/weather_final.csv', index=True)
energy.to_csv('data/energy_final.csv', index=True)
data.to_csv('data/data_final.csv', index=True)