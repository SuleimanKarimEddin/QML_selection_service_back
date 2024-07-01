<?php

namespace App\Http\Controllers;

use App\Http\Requests\SelectionRequest;
use App\Models\User;
use Illuminate\Support\Facades\Storage;

class SelectionController extends Controller
{
    public function uploadCsv(SelectionRequest $request)
    {
        $user = $request->user();
        if ($user->attemps === 0) {
            return response()->json(['message' => 'No attempts left'], 403);
        }
        $path = $request->file('file')->store('uploads');
        $fullPath = Storage::path($path);

        [$rows, $columns] = $this->getCsvDimensions($fullPath);

        User::where(['id' => $user->id])->update(['attemps' => $user->attemps - 1]);

        return response()->json(['rows' => $rows, 'columns' => $columns]);
    }

    private function getCsvDimensions($filePath)
    {
        $file = fopen($filePath, 'r');
        $rowCount = 0;
        $columnCount = 0;

        if ($file !== false) {
            while (($data = fgetcsv($file)) !== false) {
                $rowCount++;
                if ($rowCount === 1) {
                    $columnCount = count($data);
                }
            }
            fclose($file);
        }

        return [$rowCount, $columnCount];
    }
}
