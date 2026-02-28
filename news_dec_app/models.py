from django.db import models

class NewsPrediction(models.Model):
    text = models.TextField()
    prediction = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField(null = True, blank= True)

    def __str__(self):
        return f"{self.prediction} - {self.created_at}"
