from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .search_engine import SearchEngine
from django.core.paginator import Paginator
from .models import Recipes
from django.contrib.sessions.backends.db import SessionStore
import time
from django.views.decorators.csrf import csrf_protect

searchEngine = SearchEngine()
searchEngine.load_inverse_index()


# Create your views here.
def search(request):
    recipe_ids = request.session.get('stored_content')
    if request.method == 'POST':
        query = request.POST['query'].split()
        recipe_ids = searchEngine.search_by_ingredients(query, len(query))
        items_list = Recipes.objects.filter(id__in=recipe_ids)
        page_number = 1
    elif recipe_ids is not None:
        items_list = Recipes.objects.filter(id__in=recipe_ids)
        page_number = request.GET.get('page')
    else:
        items_list = Recipes.objects.all()
        page_number = request.GET.get('page')
    paginator = Paginator(items_list, 30)
    page_obj = paginator.get_page(page_number)
    data = [int(x) for x in recipe_ids]
    request.session['stored_content'] = data
    return render(request, 'index.html', {'page_obj': page_obj, "stored_content": data})
