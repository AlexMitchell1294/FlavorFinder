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

# user_id_label_reverse
top_recipes = [263566, 460763, 389536, 33311, 144686, 111724, 297142, 292451, 309168, 317784, 313902, 372915]
# top_recipes = [user_id_label_reverse[r] for r in top_recipes]

# Create your views here.
def search(request):
    print(request.user.username)
    recipe_ids = request.session.get('stored_content')
    minimum = 0 if request.session.get('minimum') is None else request.session.get('minimum')
    maximum = 120 if request.session.get('maximum') is None else request.session.get('maximum')
    last_query = "" if request.session.get('last_query') is None else request.session.get('last_query')
    if request.method == 'POST':
        amount = request.POST['amount'].split("-")
        minimum = int(amount[0])
        maximum = int(re.sub("[^0-9]", "", amount[1]))
        query = request.POST['query']
        last_query = query
        recipe_ids = text_search(query, request.user)
        algo = request.POST['algo']
        if algo == "nn":
            print("LOG: NN selected")
            items_list = sort_recipes(recipe_ids, request.user.username, minimum, maximum)
            page_number = 1
        # elif algo == "user_col":
        #     print("LOG: user collaborative filter selected")
        #     page_number = 1
        #     pass
        elif algo == "item_col":
            print("LOG: item collaborative filter selected")
            cfPredictions = [predictRatingCosine(request.user.username, recipe_ids[i]) for i in range(len(recipe_ids))]
            pred_ratings = [predict_rating(i[0], i[1]) for i in cfPredictions]
            s = sorted(zip(pred_ratings, recipe_ids))
            ranked_ids, sorted_list2 = zip(*s)
            whens = [When(id=id_val, then=pos) for pos, id_val in enumerate(ranked_ids)]

            # Create a Case expression using the When expressions
            case_expression = Case(*whens, default=0, output_field=IntegerField())
            items_list = Recipes.objects.filter(id__in=ranked_ids, minutes__range=(minimum, maximum)).order_by(case_expression)
            page_number = 1
            pass
        else:
            print("bad request")
    elif recipe_ids is not None and recipe_ids != []:
        items_list = sort_recipes(list(recipe_ids), request.user.username, minimum, maximum)
        page_number = request.GET.get('page')
    else:
        page_number = request.GET.get('page')
        items_list = Recipes.objects.all()[:1000]

    if not recipe_ids:
        page_number = request.GET.get('page')
        items_list = Recipes.objects.all()[:1000]

    # request.session["recipe_ids"] = recipe_ids
    paginator = Paginator(items_list, 30)
    page_obj = paginator.get_page(page_number)
    # if recipe_ids is not None:
    #     data = [int(x) for x in recipe_ids]
    # else:
    #     data = None
    request.session['stored_content'] = recipe_ids
    request.session['minimum'] = minimum
    request.session['maximum'] = maximum
    request.session['last_query'] = last_query
    return render(request, 'index.html', {'page_obj': page_obj, "user": request.user,
                                          "last_query": last_query})


def recipe(request):
    recipe_data = Recipes.objects.get(id=request.GET.get('key'))
    steps = ast.literal_eval(recipe_data.steps)
    ingredients = ast.literal_eval(recipe_data.ingredients)
    key = request.GET.get('key')
    try:
        content_recs = similar_recipes(tf_idf_embeddings, int(key), 12)
    except:
        content_recs = top_recipes
    print(content_recs)
    # content_recs = [item_id_label_reverse[i] for i in content_recs]
    reccomended_items = Recipes.objects.filter(id__in=content_recs)
    print(reccomended_items)
    grouped = [reccomended_items[0:4], reccomended_items[4:8], reccomended_items[8:12]]
    print(grouped)

    # steps = generic.ListView(recipe_data.steps)
    return render(request, 'recipe.html',
                  {"recipe": recipe_data,
                           "steps": steps,
                           "ingredients": ingredients,
                            "grouped_items": grouped})
