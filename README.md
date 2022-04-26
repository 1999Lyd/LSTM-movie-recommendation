# LSTM-movie-recommendation
## introduction
- Implementation of LSTM on [Amazon movie review](https://data.mendeley.com/datasets/kb5nv7dbtm/1). Explore how time information will influence the recommendation system.
## Get started
- run
```python main.py```
to train the model, model will be saved in "models" folder.
## Demo application
- application is available on [Google Cloud](https://lyd990404.ue.r.appspot.com/)
- download the [dataset](https://storage.googleapis.com/lyd990404.appspot.com/allrev.csv) and [model](https://storage.googleapis.com/lyd990404.appspot.com/fullmodel.pt) to the home directory
- run the application directly by Dockerfile: 
``` 
docker build . -t app.py
docker run -p 8080:8080 app.py
```
- or run
```python app.py```
## training result
- LSTM: 0.34 MSE loss after 265 mins training
![1a1460b7377db35f608e10a159a3700](https://user-images.githubusercontent.com/87921304/165391770-9fcc4203-e38d-4684-877d-eb7baabfc1cd.jpg)
- MF: 1.115 MSE loss after 299 mins training
![8f72016873fffb3193736bb275f628b](https://user-images.githubusercontent.com/87921304/165391887-89cc6225-4594-4399-a22e-cdbb113ca20d.png)
