from django.contrib.auth import get_user_model
from django.db import models


class SearchHistory(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name="search_histories")
    query_image = models.CharField(max_length=500)
    top_k = models.PositiveIntegerField(default=10)
    results_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
