from django.db import models


# Create your models here.

class Media(models.Model):
    name = models.CharField(max_length=30)
    photo = models.FileField()

    class Meta:
        app_label = 'server.media'

    def __str__(self):
        return self.name
