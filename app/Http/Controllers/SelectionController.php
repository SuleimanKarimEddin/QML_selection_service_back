<?php

namespace App\Http\Controllers;

use App\Http\Requests\SelectionRequest;
use App\Models\Files;
use App\Models\User;
use App\service\LocalImageHelper;
use Illuminate\Support\Facades\Http;

class SelectionController extends Controller
{
    public function __construct(private LocalImageHelper $localImageHelper)
    {
        
    }
    public function uploadCsv(SelectionRequest $request)
    {
        $user = $request->user();
        if ($user->attemps === 0) {
            return response()->json(['message' => 'No attempts left'], 403);
        }
        $file = $request->file('file');
        $targetColumnName = $request->input('target_column_name');
    
        $response = Http::timeout(86400)->attach(
            'file', file_get_contents($file->getRealPath()), $file->getClientOriginalName()
        )->post('http://python-service:8000/uploadfile/', [
            'target_column_name' => $targetColumnName,
        ]);
    
        if ($response->successful()) {
            $pdfContent = $response->body();
            $pdfPath = $this->localImageHelper->saveRawFile($pdfContent,'pdf');
            User::where(['id' => $user->id])->update(['attemps' => $user->attemps - 1]);
            Files::create([
                'user_id' => $user->id,
                'url' => $pdfPath,
            ]);
            return response()->json($pdfPath);
        }
    
        return response()->json(['error' => 'Failed to generate report'], 500);
    }
    public function userAttemps(){
        $user = auth()->user();
        return response()->json($user->attemps);
    }
    public function test(SelectionRequest $request){
        $file = $request->file('file');
        $targetColumnName = $request->input('target_column_name');
    
        $response = Http::attach(
            'file', file_get_contents($file->getRealPath()), $file->getClientOriginalName()
        )->post('http://127.0.0.1:8000/pdf/', [
            'target_column_name' => $targetColumnName,
        ]);

        if ($response->successful()) {
            $pdfContent = $response->body();
            $pdfPath = $this->localImageHelper->saveRawFile($pdfContent,'pdf');
            return response()->json($pdfPath);
        }

        return response()->json(['error' => 'Failed to generate report'], 500);

    }

 
    private function savePdfToFile($pdfContent)
    {
        // Define the path where the PDF should be saved
        $filePath = storage_path('app/public/') . 'downloaded_file.pdf';
    
        // Save the content to the file
        file_put_contents($filePath, $pdfContent);
    
        return $filePath;
    }
  
}
