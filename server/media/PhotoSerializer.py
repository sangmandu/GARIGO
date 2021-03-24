from rest_framework import serializers

from media.models import Media


class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        app_label = 'server.media'

        model = Media
        fields = ('__all__')
