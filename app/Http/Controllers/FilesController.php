<?php

namespace App\Http\Controllers;

use App\Http\Requests\Files\DeleteUserFile;
use App\Http\Requests\Files\UploadUserFile;
use App\Models\Files;
use App\service\LocalImageHelper;
use Illuminate\Http\Request;

class FilesController extends Controller
{
    public function __construct(private LocalImageHelper $localImageHelper) {}

    public function getAllFiles()
    {
        $user = auth()->user();
        $files = Files::where('user_id', $user->id)->get();

        return response()->json($files);
    }

    public function upload(UploadUserFile $request)
    {
        $user = auth()->user();
        $url = $this->localImageHelper->saveImage($request->file);
        $files = Files::create([
            'user_id' => $user->id,
            'url' => $url,
        ]);

        return response()->json($files);
    }

    public function delete(DeleteUserFile $request)
    {
        $user = auth()->user();
        $this->localImageHelper->DeleteImage($request->url);
        Files::where('user_id', $user->id)->where('url', $request->url)->delete();
    }
}
