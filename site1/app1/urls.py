from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('register', views.register, name="register"),
    path('login', views.loginPage, name="login"),
    path('logout', views.logoutPage, name="logout"),
    path('change_password', views.change_password, name='change_password'),
    path('drowsiness/', views.drowsiness, name="drowsiness"),
]