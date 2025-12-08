package ru.happines;

import java.awt.*;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiPredicate;

public class second_lab {
    public static void main(String[] args) throws Exception {
        int size = 100;
        int target = 42;

        BiPredicate<Integer, Integer> containsNumber = (n, x) -> {
            Random rand = new Random();
            int[] arr = new int[n];

            for (int i = 0; i < n; i++) {
                arr[i] = rand.nextInt(100);
            }

            System.out.println("Сгенерированный массив: " + Arrays.toString(arr));
            return Arrays.stream(arr).anyMatch(i -> i == x);
        };

        ExecutorService executor = Executors.newSingleThreadExecutor();

        Callable<Boolean> task = () -> containsNumber.test(size, target);
        Future<Boolean> future = executor.submit(task);

        while (!future.isDone()) {
            System.out.println('.');
            Thread.sleep(100);
        }

        boolean result = future.get();

        executor.shutdown();

        System.out.println("\nЧисло " + target+ (result ? " найдено" : " не найдено") + " в массиве");


    }


}
