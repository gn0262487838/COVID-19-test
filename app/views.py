from django.shortcuts import render
from django.http import HttpResponse
from html import unescape
from .models import UserImage
from covid19_nn._predict import predictCovid19
import os


def index(request):

    if request.method == "GET":

        return render(request, "index.html", locals())


def upload(request):

    if request.method == "POST":

        user_image = request.FILES.get("img")

        if user_image is None:
            response = HttpResponse('Unkown File')
            response.status_code = 400
            return response

        data = UserImage(user_image=user_image)

        if not os.path.splitext(str(data.user_image.url))[-1].lower() in [".jpeg", ".jpg", ".png"]:
            return render(request, "error.html")

        data.save()
        pred = predictCovid19(user_image)

        return render(request, "upload.html", locals())


def contact(request):

    if request.method == "GET":
        return render(request, "contact.html", locals())
    

