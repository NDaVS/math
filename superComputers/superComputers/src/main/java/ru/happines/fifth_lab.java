package ru.happines;

import java.util.Random;

public class fifth_lab {
    public static void coolMatrix(int rows, int cols, int numThread) {
        double [][] matrix = new double[rows][cols];
        Random random = new Random();

        for (int i = 0 ; i < rows; i ++) {
            for (int j =0; j < cols ;j++) {
                matrix[i][j] = random.nextDouble() * 10;
            }
        }

        Thread[] threads = new Thread[numThread];

        int rowsPerThread = rows / numThread;

        for (int t = 0; t < numThread; t ++) {
            final int startRow = t * rowsPerThread;
            final int endRow = (t == numThread - 1) ? rows: startRow + rowsPerThread;

            threads[t] = new Thread(() -> {
                for (int i = 0 ; i < rows; i ++) {
                    for (int j = 0; j < cols; j++) {
                        matrix[i][j] = Math.cos(matrix[i][j]);
                    }
                }
            });
            threads[t].start();
        }

        for (Thread thread: threads) {
            try{
                thread.join();
            } catch (InterruptedException e){
                e.printStackTrace();
            }
        }

//        System.out.println("Преобразованная матрица:");
//        for (double[] row : matrix) {
//            for (double val : row) {
//                System.out.printf("%.3f ", val);
//            }
//            System.out.println();
//        }
    }

    public static void main(String[] args) {
        int rows = 5000;
        int cols = 5000;

        int numThreads = 1;
        long startTime = System.nanoTime();

        coolMatrix(rows, cols, numThreads);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        double milliseconds = duration / 1_000_000.0;

        System.out.printf("Время выполнения: %.3f мс%n", milliseconds);
    }
}
