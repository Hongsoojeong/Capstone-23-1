from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


# Create your views here.
def index(request):
    template = loader.get_template("polls/index.html")
    return render(request, "polls/index.html")


def analysis(request):
    template = loader.get_template("polls/analysis.html")
    return render(request, "polls/analysis.html")


def login(request):
    template = loader.get_template("polls/login.html")
    return render(request, "polls/login.html")


def mypage(request):
    template = loader.get_template("polls/mypage.html")
    return render(request, "polls/mypage.html")


def signUp(request):
    template = loader.get_template("polls/signUp.html")
    return render(request, "polls/signUP.html")


def recording(request):
    template = loader.get_template("polls/recording.html")
    return render(request, "polls/recording.html")
