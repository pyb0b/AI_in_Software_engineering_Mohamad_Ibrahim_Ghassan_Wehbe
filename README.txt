# Special recommender system:
recommend a movie for a user knowing what movies he have rated and what others have rated (collaborative filtering)
all while taking into consideration MLops factors

## Dependencies:
numpy
pandas
scikit-learn
fastapi
prometheus-fastapi-instrumentator
requests

pip install <dependency name>


## Usage
to run this project first navigate to the directory, then type python main.py.
next, run app.py using the following command on the cmd:
uvicorn app:app --reload
you can use three endpoints with three different endpoints using the POST method on Postman.
The inputs form should be raw and are indicated in the PowerPoint for each endpoint.
The step for running docker are indicated in the PowerPoint also.
you can try test_gui.py if the app.py is running. type python test_gui.py
