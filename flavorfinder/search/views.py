from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .search_engine import *
from django.core.paginator import Paginator
from .models import Recipes
from django.db.models import When, Case, IntegerField
from django.contrib.auth.models import User
from django.contrib.sessions.backends.db import SessionStore
import time
from django.views.decorators.csrf import csrf_protect
from tensorflow.keras.models import load_model
from django.views import generic
import ast


# Create your views here.
def search(request):
    print(request.user.username)
    recipe_ids = request.session.get('stored_content')
    if request.method == 'POST':
        query = request.POST['query']
        recipe_ids = text_search(query, request.user)

        items_list = sort_recipes(recipe_ids, request.user.username)
        page_number = 1
    elif recipe_ids is not None:
        items_list = sort_recipes(list(recipe_ids), request.user.username)
        page_number = request.GET.get('page')
    else:
        page_number = request.GET.get('page')
        items_list = Recipes.objects.all()[:1000]

    # request.session["recipe_ids"] = recipe_ids
    paginator = Paginator(items_list, 30)
    page_obj = paginator.get_page(page_number)
    if recipe_ids is not None:
        data = [int(x) for x in recipe_ids]
    else:
        data = None
    request.session['stored_content'] = data
    return render(request, 'index.html', {'page_obj': page_obj, "stored_content": data, "user": request.user})


def recipe(request):
    recipe_data = Recipes.objects.get(id=request.GET.get('key'))
    steps = ast.literal_eval(recipe_data.steps)
    ingredients = ast.literal_eval(recipe_data.ingredients)
    reccomended_items = Recipes.objects.all()[:12]
    grouped = [reccomended_items[0:4], reccomended_items[4:8], reccomended_items[8:12]]

    # steps = generic.ListView(recipe_data.steps)
    return render(request, 'recipe.html',
                  {"recipe": recipe_data,
                           "steps": steps,
                           "ingredients": ingredients,
                            "grouped_items": grouped})
