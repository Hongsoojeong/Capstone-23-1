from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import VoiceRecording
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
from datetime import datetime  # datetime 모듈 추가


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


def mypage(request):
    if request.method == "POST":
        recording = VoiceRecording(audio_file=request.FILES["audio_file"])
        recording.save()
        return JsonResponse(
            {
                "id": recording.id,
                "uploaded_at": recording.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    else:
        recordings = VoiceRecording.objects.all()
        context = {"recordings": recordings}
        return render(request, "polls/mypage.html", context)


def recording(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio_file")
        if audio_file:
            # 파일이 올바르게 첨부된 경우
            # 파일을 읽어들이고 데이터베이스에 저장
            file_name = default_storage.save(
                audio_file.name, ContentFile(audio_file.read())
            )
            recording = VoiceRecording(audio_file=file_name)
            recording.save()
            return JsonResponse(
                {
                    "id": recording.id,
                    "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    return render(request, "polls/recording.html")
