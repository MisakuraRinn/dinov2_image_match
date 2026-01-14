from django.urls import path
from . import views

app_name = "retrieval"

urlpatterns = [
    path("", views.home, name="home"),
    path("search/", views.search, name="search"),
    path("history/", views.history, name="history"),
    path("history/delete/<int:entry_id>/", views.delete_history, name="delete_history"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("gallery/<path:relpath>/", views.gallery_image, name="gallery"),
]
