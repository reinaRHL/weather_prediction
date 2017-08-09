Project has 3 parts. 

Run each program using these command lines:


1. Weather label prediction program
	
	python3 weather.py yvr-weather katkam-scaled_200 testing_img 


2. Height of tide and Time of the day prediction program

	python3 tide.py tideData.csv katkam-scaled_200 testing_img


3. Weather forcase prediction program
	
	python3 forecast.py yvr-weather image_at_16 image_at_17


Note:
	#1 takes at least 5-6 min.
	#2 takes at least 1-2 min.
	#3 takes at least 2-3 min.


They all print accuracy score for each model and predicted label for the sample testing input.

Expected outputs for #1
    Accuracy score of the model:
        Accuracy score for weather label prediction:
		0.xx

    Prediction values from the sample input: (actual value can be different)
	  	Weather Label Prediction from the sample input:
		[('Cloudy',), ('Cloudy',), ('Cloudy',), ('Clear',), ('Clear',), ('Drizzle',), ('Rain', 'Fog'),('Snow',)]


Expected outputs for #2
    Accuracy score of the model
        Accuracy score for tide height prediction:
		0.xx
		Accuracy score for time prediction:
		0.xx

    Prediction values from the sample input. (actual output values can be different)
	  	Tide height prediction from the sample input:
		['low' 'low' 'low' 'low' 'low' 'high' 'high' 'medium']

		Time prediction from the sample input:
		['afternoon' 'afternoon' 'afternoon' 'afternoon' 'afternoon' 'afternoon' 'afternoon' 'afternoon']

Expected outputs for #3
    Accuracy score of the model
        Accuracy score for weather prediction:
		0.xx

    Weather Forecast prediction from the sample input:
		[ 0.  0.  0.  0.  1.  1.  0.  0.  1.  0.  1.]

    0 means it will not rain/snow tomorrow ----  1 means it will rain/snow tomorrow




