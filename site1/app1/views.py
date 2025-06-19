from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.decorators import login_required

# Create your views here.
def home(request):
    return render(request,'app1/home.html') 

def register (request):
    form = UserCreationForm()
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
        
    context={'form':form}
    return render(request,'app1/register.html',context)

def loginPage (request):
    # if request.user.is_authenticated:
    #     return redirect('home')
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request,username = username, password = password)
        if user is not None:
            login(request,user)
            return redirect('home')
        else: messages.info(request,'user or password not correct!')
    context={}
    return render(request,'app1/login.html',context)

def logoutPage(request):
    logout(request)
    return redirect('login')

def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password was successfully updated!')
            return redirect('home')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'app1/change_password.html', {
        'form': form
    })

@login_required
def drowsiness(request):
    return render(request, 'app1/drowsiness.html')