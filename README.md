# COVID-19 Only Academic Use

## Install on Ubuntu(AWS)

### step 1 git clone

### step 2 sudo apt-get update

### step 3 sudo apt-get install python3-pip

### step 4 sudo apt-get install virtualenv

### step 5 cd "your project"

### step 6 virtualenv -p /usr/bin/python3.6 "your env name"

### step 7 source /your env name/bin/activate

### step 8 which pip3

### step 9 pip3 install -r requirement

### step 10 cd COVID-19-test/app/ ; mkdir Models  *remember upload your model

### step 11 python manage.py makemigrations

### step 12 python manage.py migrate

### step 13 python manage.py runserver ip:port --insecure


