# LSTM-movie-recommendation
## Get started
- download the [dataset](https://storage.googleapis.com/lyd990404.appspot.com/allrev.csv) and [model](https://storage.googleapis.com/lyd990404.appspot.com/fullmodel.pt) to the home directory
- run the application directly by Dockerfile: 
``` 
docker build . -t app.py
docker run -p 8080:8080 app.py
```
- or run
```python app.py```
