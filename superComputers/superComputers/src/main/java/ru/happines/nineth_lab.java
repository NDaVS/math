package ru.happines;

import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class nineth_lab {

    public static int[][] generateMatrix() {
        Random random = new Random();
        int rows = random.nextInt(4) + 3;
        int cols = random.nextInt(4) + 3;

        int[][] matrix = new int[rows][cols];

        System.out.println("Матрица имеет размер (" + rows + 'x' + cols + "):");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextInt(100);
                System.out.printf("%4d", matrix[i][j]);
            }

            System.out.println();
        }

        System.out.println();

        return matrix;
    }

    public static int sumEvenElements(int[][] matrix) {
        int sum = 0;
        for (int[] row : matrix) {
            for (int value : row) {
                if (value % 2 == 0) {
                    sum += value;
                }
            }
        }

        return sum;
    }

    public static double avgElements(int[][] matrix) {
        int sum = 0;
        int count = 0;

        for (int[] row : matrix) {
            for (int value : row) {
                if (value % 2 == 0) {
                    sum += value;
                    count++;
                }
            }
        }
        return (double) sum / count;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        CompletableFuture<int[][]> matrixTask = CompletableFuture.supplyAsync(nineth_lab::generateMatrix);
        CompletableFuture<Integer> sumEvenTask = matrixTask.thenApplyAsync(nineth_lab::sumEvenElements);
        CompletableFuture<Double> avgTask = matrixTask.thenApplyAsync(nineth_lab::avgElements);

        CompletableFuture<Void> allTasks = CompletableFuture.allOf(sumEvenTask,avgTask).thenRun(
                () -> {
                    try {
                        System.out.println("Сумма четных элементов: " + sumEvenTask.get());
                        System.out.println("Среднее арифметическое элементов: " + avgTask.get());
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
        );

        allTasks.get();
    }
}
