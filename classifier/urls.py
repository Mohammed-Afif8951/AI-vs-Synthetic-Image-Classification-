
from django.urls import path
from classifier import views
from . import views

urlpatterns = [
    
    path('', views.index, name='index'),
]
