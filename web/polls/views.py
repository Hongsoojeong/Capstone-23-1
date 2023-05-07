from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.conf import settings
import os
from .models import VoiceRecording
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


def signUp(request):
    template = loader.get_template("polls/signUp.html")
    return render(request, "polls/signUP.html")


def mypage(request):  # 여기가 문제인것 같음
    record = request.POST.get("audio_file")
    # recordings = VoiceRecording.objects.all()
    context = {"recordings": record}
    return render(request, "polls/mypage.html", context)


def recording(request):
    if request.method == "POST" and request.FILES.get("audio"):
        return handle_audio_upload(request)

    return display_recordings(request)


def handle_audio_upload(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        recording = VoiceRecording(audio_file=audio_file)

        # 파일 저장 경로와 파일명 설정
        file_path = os.path.join(settings.MEDIA_ROOT, "recordings", audio_file.name)

        with open(file_path, "wb") as f:
            f.write(audio_file.read())
            recording.audio_file.name = os.path.relpath(f.name, settings.MEDIA_ROOT)

        recording.save()

        return HttpResponse("ok")

    return HttpResponseBadRequest("Invalid Request: 'audio' file not found")


def display_recordings(request):
    recordings = VoiceRecording.objects.all()
    context = {"recordings": recordings}
    return render(request, "polls/recording.html", context)
