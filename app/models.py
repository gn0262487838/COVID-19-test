from django.db import models
import datetime

class UserImage(models.Model):
    
    user_image = models.ImageField(upload_to=f"images/{datetime.datetime.today().strftime('%Y_%m_%d')}")
