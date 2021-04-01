import uuid
from io import BytesIO
import os

import boto3
from PIL import Image
from django.http import JsonResponse
from requests import Response
from rest_framework import status
from rest_framework.views import APIView
from .mosaic import mosaic_video
from .make_audio import make_audio

from media.PhotoSerializer import PhotoSerializer
from util.MongoManager import createMedia, getImageByPid
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile



class ProfileUploadView(APIView):
    def __init__(self):
        s3_client = boto3.client(
                's3',
                aws_access_key_id="AKIAR26SC7FY7ESCGFCB",
                aws_secret_access_key="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",
            )
    def get(self, request):
        ROOT_DIR = os.getcwd()
        path = os.path.join(ROOT_DIR, 'input_video/')
        s3_client = boto3.client(
                's3',
                aws_access_key_id="AKIAR26SC7FY7ESCGFCB",
                aws_secret_access_key="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",
            )
        root_url = "https://garigo.s3.ap-northeast-2.amazonaws.com/"
        raw_recog_image_set = getImageByPid('c4a63351-f8b1-45bf-9c7e-ac2934b33d95')

        # recognition image를 받아옴
        recog_list = list(raw_recog_image_set.keys())
        recog_images = []
        for name in recog_list:
            img_urls = []
            for img_list in raw_recog_image_set[name]:
                img_url = root_url + img_list["fileUid"]
                img_urls.append(img_url)
            recog_images.append(img_urls)
        

        # input_video = "https://garigo.s3.ap-northeast-2.amazonaws.com/clip.mp4"

        # upload된 video를 저장
        input_video = request.FILES.getlist('photos')[0]
        filename = str(input_video)
        input_route = os.path.join(path, filename)
        with open(input_route, 'wb+') as f:
            for chunk in input_video.chunks():
                f.write(chunk)



        # video처리 후 저장 된 video를 서버에 올려줌
        output_route = mosaic_video(recog_list, recog_images, input_route, filename)
        path = os.path.join(ROOT_DIR, 'output_video/')
        output_route = os.path.join(path, "output_stop.mp4")

        final_route = make_audio(input_route, output_route, filename)

        # output_video = "/Users/sumin/project/programmers/GARIGO/server/output_video/output_input.mp4"
        with open(final_route, 'rb') as data:
            pid = str(uuid.uuid4())
            name, extend = final_route.split(".")
            fileUid = f'{str(uuid.uuid4())}.{extend}'
            s3_client.put_object(
                        Body=data,
                        Bucket="garigo",
                        Key=fileUid,
                    )

        root_url = "https://garigo.s3.ap-northeast-2.amazonaws.com/"
        raw_recog_image_set = getImageByPid('c4a63351-f8b1-45bf-9c7e-ac2934b33d95')
        recog_list = list(raw_recog_image_set.keys())
        recog_images = []
        for name in recog_list:
            img_urls = []
            for img_list in raw_recog_image_set[name]:
                img_url = root_url + img_list["fileUid"]
                img_urls.append(img_url)
            recog_images.append(img_urls)
        input_video = ""
        output_video = mosaic_video(recog_list, recog_images, input_video)
        # print()

        return JsonResponse({'a': 'a'})

    def post(self, request):
        if len(request.FILES) != 0:
            s3_client = boto3.client(
                's3',
                aws_access_key_id="AKIAR26SC7FY7ESCGFCB",
                aws_secret_access_key="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",
            )

            pid = str(uuid.uuid4())

            for file in request.FILES.getlist('photos'):
                name, extend = file.name.split(".")
                fileUid = f'{str(uuid.uuid4())}.{extend}'
                s3_client.put_object(
                    Body=file,
                    Bucket="garigo",
                    Key=fileUid,
                )

                createMedia(pid, name.split("_")[0], name.split("_")[1], fileUid)
            return JsonResponse({'pid': pid})
        else:
            return JsonResponse({'message': 'file_none'})

