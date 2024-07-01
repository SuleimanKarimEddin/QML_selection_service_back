<?php

namespace App\Http\Controllers;

use App\Http\Requests\User\LoginRequest;
use App\Http\Requests\User\RegisterRequest;
use App\service\UserService;
use Illuminate\Http\Request;

class UserController extends Controller
{
    public function __construct(private UserService $userService)
    {
        
    }

    public function register(RegisterRequest $request){
        $data = $this->userService->register($request->validated());
        return response()->json($data);
    }

    public function login(LoginRequest $request){
        $data =$this->userService->login($request->validated());
        return response()->json($data);
    }


    public function logout(Request $request){
        $request->user()->currentAccessToken()->delete();
    }
}
