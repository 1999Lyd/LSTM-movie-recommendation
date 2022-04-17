FROM public.ecr.aws/lambda/python:3.7

RUN mkdir -p /app
COPY . app.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD streamlit run app.py
ENTRYPOINT [ "python" ]
