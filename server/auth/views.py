import jwt
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView

# Create your views here.
from server.jwt_secret import SECRET_KEY, ALGORITHM
from util import MongoManager


class AuthView(APIView):
    def get(self, request):
        id = request.query_params['id']
        pw = request.query_params['pw']

        user = MongoManager.getUser(id, pw)

        if user:
            payload = {
                'id': user['id']
            }

            token = jwt.encode(payload, SECRET_KEY, ALGORITHM)
            return HttpResponse(token, status=200)
        return HttpResponse(status=401)

    def post(self, request):
        id = request.data['id']
        pw = request.data['pw']
        _id = MongoManager.createUser(id, pw)
        return HttpResponse(status=201)
