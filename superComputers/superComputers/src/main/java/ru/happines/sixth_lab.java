package ru.happines;

import java.util.Random;

class MaxFinder extends Thread {
    private int size1;
    private int size2;

    MaxFinder(int size1, int size2) {
        this.size1 = size1;
        this.size2 = size2;
    }

    @Override
    public void run() {
        int[] arr_1 = new int[size1];
        int[] arr_2 = new int[size2];
        Random rand = new Random();

        for (int i = 0; i < size1; i++) {
            arr_1[i] = rand.nextInt(1000);
        }
        for (int i = 0; i < size2; i++) {
            arr_2[i] = rand.nextInt(1000);
        }

        int max_1 = Integer.MIN_VALUE;
        int max_2 = Integer.MIN_VALUE;

        for (int num : arr_1) {
            if (num > max_1) {
                max_1 = num;
            }
        }

        for (int num : arr_2) {
            if (num > max_2) {
                max_2 = num;
            }
        }
        System.out.println("Для массива размера " + size1 + ":  максимум = " + max_1);
        System.out.println("Для массива размера " + size2 + ":  максимум = " + max_2);
    }
}

public class sixth_lab {
    public static void main(String[] args) {
        int n1 = 10;
        int n2 = 15;

        Thread t1 = new MaxFinder(n1, n2);

        t1.start();

        try {
            t1.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        System.out.println("Вычисления завершены");
    }
}
