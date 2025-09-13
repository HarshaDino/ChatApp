from django.urls import path
from .views import UploadView, ChatView, index

urlpatterns = [
    path('upload/', UploadView.as_view(), name='upload'),
    path('chat/', ChatView.as_view(), name='chat'),
]
