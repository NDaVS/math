package ru.happines;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;
import java.util.function.BiPredicate;

public class third_lab {
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
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

        Callable<Boolean> task = () -> {
//            Thread.sleep(3000);
            return containsNumber.test(size, target);
        };
        Future<Boolean> future = executor.submit(task);

        scheduler.schedule(() -> {
            if (!future.isDone()) {
                System.out.println("Отмена по таймеру");
                future.cancel(true);
            }
        }, 1, TimeUnit.SECONDS);

        try {
            boolean result = future.get();
            System.out.println("\nЧисло " + target+ (result ? " найдено" : " не найдено") + " в массиве");
        } catch (CancellationException e) {
            System.out.println("Задача была отменена");
        } finally {
            executor.shutdown();
            scheduler.shutdownNow();
        }
    }
}
