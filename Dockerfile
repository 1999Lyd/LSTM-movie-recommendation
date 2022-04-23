FROM python:3.7

# Create working directory
WORKDIR /app
ADD https://storage.googleapis.com/lyd990404.appspot.com/allrev.csv /app
ADD https://storage.googleapis.com/lyd990404.appspot.com/fullmodel.pt /app
# Copy requirements.txt file
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# Copy all files for app
COPY . /app

#Run app
CMD ["gunicorn", "app:app", "-b", ":8080", "--timeout", "300"]
