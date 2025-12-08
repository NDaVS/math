package ru.happines;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class eighth_lab {
    public static long factorial(long n) {
        if (n < 0)
            throw new IllegalArgumentException("Работаем только с положительными числами");

        long result = 1;

        for (long i = 2; i <= n; i++) {
            result *= i;
        }

        return result;
    }

    public static CompletableFuture<Long> factorialAsync(long n) {
        return CompletableFuture.supplyAsync(() -> factorial(n));
    }

    public static void main(String[] args) {
        try {
            long number = 10;
            CompletableFuture<Long> task = factorialAsync(number);

            System.out.println("Вычисляется факториал для " + number + "...");
            long result = task.get();
            System.out.println("Факториал числа равен " + result);
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
