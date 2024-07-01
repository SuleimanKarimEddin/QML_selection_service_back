<?php

namespace App\service;

use App\Models\User;
use Illuminate\Support\Facades\Hash;

class UserService
{
    public function __construct() {}

    public function register(array $data)
    {
        $data['password'] = Hash::make($data['password']);
        $user = User::create($data);

        return [
            'user' => $user,
            'token' => $user->createToken('auth_token')->plainTextToken,
        ];
    }
    public function login(array $data){
        $user = User::where('email', $data['email'])->first();
        if (!$user || !Hash::check($data['password'], $user->password)) {
            throw new \Exception('Login failed');
        }

        return [
            'user' => $user,
            'token' => $user->createToken('auth_token')->plainTextToken,
        ];
    }

}
