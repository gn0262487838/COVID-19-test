from django.shortcuts import render
from django.http import HttpResponse
from .models import UserImage
import PIL

def index(request):

    if request.method == "GET":
        
        return render(request, "./index.html", locals())

    return HttpResponse("請求失敗")


def upload(request):

    if request.method == "POST":
        user_image = request.FILES.get("img")
        data = UserImage(user_image=user_image)
        data.save()
        
        return render(request, "./upload.html", locals())

    return HttpResponse("請求方式不正確")
