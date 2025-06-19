# routing.py (tạo file mới trong app của bạn)
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/drowsiness/$', consumers.DrowsinessConsumer.as_asgi()),
]