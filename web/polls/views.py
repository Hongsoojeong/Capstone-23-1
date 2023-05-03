from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


# Create your views here.
def index(request):
    template = loader.get_template("polls/index.html")
    return render(request, "polls/index.html")

# def analysis(request):
#     return HttpResponse("analysis")

# def login(request):
#     return HttpResponse("login")

# def mypage(request):
#     return HttpResponse("mypage")

# def signUp(request):
#     return HttpResponse("signup")

# def recording(request):
#     return HttpResponse("recording")
