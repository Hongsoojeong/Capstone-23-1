from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

app_name = "polls"
urlpatterns = [
    path("", views.index, name="index"),
    path("analysis/", views.analysis, name="analysis"),
    path("login/", views.login, name="login"),
    path("mypage/", views.mypage, name="mypage"),
    path("recording/", views.recording, name="recording"),
    path("signUp/", views.signUp, name="signUp"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
