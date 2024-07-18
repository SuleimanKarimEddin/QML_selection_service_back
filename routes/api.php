<?php

use App\Http\Controllers\FilesController;
use App\Http\Controllers\PaymentController;
use App\Http\Controllers\SelectionController;
use App\Http\Controllers\UserController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return 'heello';
});

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::prefix('auth')->controller(UserController::class)->group(function () {
    Route::post('login', 'login');
    Route::post('register', 'register');
    Route::post('logout', 'logout')->middleware('auth:sanctum');
});

Route::controller(PaymentController::class)->prefix('payment')->group(function () {

    Route::middleware(['auth:sanctum'])->post('/checkout', 'checkout');
    Route::get('/success', 'success')->name('payment.success');
    Route::get('/cancel', 'cancel')->name('payment.cancel');
});
// upload csv file
Route::middleware('auth:sanctum')->post('/uploadCsv', [SelectionController::class, 'uploadCsv']);
Route::post('/pdf', [SelectionController::class, 'test']);


Route::middleware('auth:sanctum')->controller(FilesController::class)->group(function () {

    Route::get('/getAllFiles', 'getAllFiles');
    Route::post('/delete', 'delete');
    Route::post('/upload', 'upload');
});
