<?php

namespace App\Http\Controllers;

use App\Http\Requests\Payment\CheckoutRequest;
use App\Models\Payment;
use App\Models\User;
use Illuminate\Http\Request;
use Stripe\Checkout\Session;
use Stripe\Stripe;

class PaymentController extends Controller
{
    public function checkout(CheckoutRequest $request)
    {
        Stripe::setApiKey('sk_test_51Nc5XrDFWjkjBINJkclqmIhxxWa4XzRB5etU8T0sXadZGfXt5YnjIbqB2eOp33QmJtJrIIQpPWD41TSHAp3g71M800o3Fr12sw');

        $user = auth()->user();

        $lineItems[] = [
            'price_data' => [
                'currency' => 'usd',
                'product_data' => [
                    'name' => 'QML subscription',
                ],
                'unit_amount' => $request->price,
            ],
            'quantity' => $request->attemps,
        ];
        $session = Session::create([
            'line_items' => $lineItems,
            'mode' => 'payment',
            'success_url' => route('payment.success', [], true).'?session_id={CHECKOUT_SESSION_ID}',
            'cancel_url' => route('payment.cancel', [], true).'?session_id={CHECKOUT_SESSION_ID}',
        ]);
        Payment::create([
            'user_id' => $user->id,
            'payment_id' => $session->id,
            'status' => 'pending',
            'attemps' => $request->attemps,
            'price' => $request->price,
        ]);

        return $session->url;
    }

    public function success(Request $request)
    {
        $payment_id = $request['session_id'];
        if (! $payment_id) {
            throw new \Exception('Payment id not found');
        }
        $payment = Payment::where('payment_id', $payment_id)->first();
        if (! $payment) {
            throw new \Exception('payment not found');
        }
        if ($payment->status == 'pending') {
            $user = User::where('id', $payment->user_id)->first();
            if (! $user) {
                throw new \Exception('user not found');
            }
            $user->update([
                'attemps' => $payment->attemps + $user->attemps,
            ]);
            $payment->update([
                'status' => 'paid',
            ]);
        }

        return redirect('https://google.com');
    }

    public function cancel(Request $request)
    {
        $payment_id = $request['session_id'];
        $payment = Payment::where('payment_id', $payment_id)->first();
        $payment->update([
            'status' => 'cancel',
        ]);

        return redirect('https://youtube.com');
    }
}
