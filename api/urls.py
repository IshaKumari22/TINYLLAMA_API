from django.urls import path
from .views import TinyLlamaView

urlpatterns = [
    path("generate/", TinyLlamaView.as_view(), name="generate"),
]
