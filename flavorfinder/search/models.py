from django.db import models


class Recipes(models.Model):
    index = models.IntegerField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    id = models.IntegerField(blank=True, primary_key=True)
    minutes = models.IntegerField(blank=True, null=True)
    contributor_id = models.IntegerField(blank=True, null=True)
    submitted = models.TextField(blank=True, null=True)
    tags = models.TextField(blank=True, null=True)
    nutrition = models.TextField(blank=True, null=True)
    n_steps = models.IntegerField(blank=True, null=True)
    steps = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    n_ingredients = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recipes'
