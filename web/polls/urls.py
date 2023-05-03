from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("analysis/", views.analysis, name="analysis"),
    path("login/", views.login, name="login"),
    path("mypage/", views.mypage, name="mypage"),
    path("recording/", views.recording, name="recording"),
    path("signUp/", views.signUp, name="signUp"),
]
