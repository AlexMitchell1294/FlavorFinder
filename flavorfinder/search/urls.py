from django.urls import path, include, re_path
from django.views.generic.base import RedirectView
from . import views
from django.contrib import admin

urlpatterns = [
    path('', RedirectView.as_view(url='search/'), name='index'),
    path('search/', views.search, name='search'),
    path('accounts/', include("django.contrib.auth.urls")),
    path("search/recipe/", views.recipe, name="recipe")
    # path('', include("django.contrib.auth.urls"))
]