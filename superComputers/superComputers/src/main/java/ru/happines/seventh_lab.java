package ru.happines;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class RandomNumberArray {
    private final int[] numbers;

    public RandomNumberArray() {
        Random rand = new Random();
        int size = rand.nextInt(11) + 10;

        numbers = new int[size];

        for (int i = 0; i < size; i++) {
            numbers[i] = rand.nextInt(99) + 1;
        }
    }

    public int[] getNumbers() {
        return numbers;
    }
}

class CollectionProcessor {
    private int[] numbers;
    private int delay;

    public CollectionProcessor(int[] numbers, int delayMilliseconds) {
        this.numbers = numbers;
        this.delay = delayMilliseconds;
    }

    public List<Integer> process() throws InterruptedException, ExecutionException {
        List<Integer> divisibleBy6 = Collections.synchronizedList(new ArrayList<>());
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<?>> futures = new ArrayList<>();

        for (int number : numbers) {
            futures.add(executor.submit(() -> {
                try {
                    Thread.sleep(delay);
                    if (number % 6 == 0) {
                        divisibleBy6.add(number);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }));
        }

        for (Future<?> f : futures) {
            f.get();
        }

        executor.shutdown();
        return divisibleBy6;
    }
}

public class seventh_lab {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        RandomNumberArray randomArray = new RandomNumberArray();
        System.out.println(LocalDate.now().atStartOfDay(ZoneId.systemDefault()).toInstant().toString());
        System.out.println("Сгенерированный массив:");
        System.out.println(Arrays.toString(randomArray.getNumbers()));

        int[] delays = {0, 100, 500};
        for (int delay : delays) {
            CollectionProcessor processor = new CollectionProcessor(randomArray.getNumbers(), delay);
            long startTime = System.currentTimeMillis();
            List<Integer> result = processor.process();
            long elapsed = System.currentTimeMillis() - startTime;

            System.out.println("\nЗадержка: " + delay + " мс");
            System.out.println("Числа, делящиеся на 6: " + result);
            System.out.println("Время обработки: " + elapsed + " мс");
        }

        System.out.println("\nВывод: с увеличением задержки программа обрабатывает элементы медленнее, но использование пула потоков позволяет распараллелить работу и минимизировать общее время обработки.");
    }
}
