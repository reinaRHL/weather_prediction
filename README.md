
Project has 3 parts. 

Run each program using these command lines:


1. Weather label prediction program
	
	python3 weather.py yvr-weather katkam-scaled_700 testing_img 


2. Height of tide and Time of the day prediction program

	python3 tide.py tideData.csv katkam-scaled_700 testing_img


3. Weather forcase prediction program
	
	python3 forecast.py yvr-weather image_at_16 image_at_17


Note:
	#1 takes at least 18-20 min.
	#2 takes at least 4-5 min.
	#3 takes at least 2-3 min.

	They all print accuracy score for each model and predicted label for the sample testing input.

	Please refer to the report what each output means in detail.
