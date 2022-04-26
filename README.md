# LSTM-movie-recommendation
## introduction
- Implementation of LSTM on [Amazon movie review](https://data.mendeley.com/datasets/kb5nv7dbtm/1). Explore how time information will influence the recommendation system.
## Get started
- run
```python main.py```
to train the model, model will be saved in "models" folder.
## demo application
- download the [dataset](https://storage.googleapis.com/lyd990404.appspot.com/allrev.csv) and [model](https://storage.googleapis.com/lyd990404.appspot.com/fullmodel.pt) to the home directory
- run the application directly by Dockerfile: 
``` 
docker build . -t app.py
docker run -p 8080:8080 app.py
```
- or run
```python app.py```
