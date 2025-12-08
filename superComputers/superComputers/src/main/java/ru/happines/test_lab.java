package ru.happines;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class test_lab {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();

        Callable<Integer> task = () -> {
            TakesAWhile(1, 3000);
            return 42;
        };
        Future<Integer> future = executor.submit(task);

        while (!future.isDone()) {
            System.out.println('.');
            Thread.sleep(50);
        }

        int result = future.get();
        System.out.println("\nresult: " + result);

        executor.shutdown();
    }

    static void TakesAWhile(int a, int delay) throws InterruptedException {
        System.out.println("Working...");
        Thread.sleep(delay);
        System.out.println("Done!");
    }
}
