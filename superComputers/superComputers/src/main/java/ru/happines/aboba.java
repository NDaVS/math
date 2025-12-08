package ru.happines;

import java.util.function.Function;

@FunctionalInterface
interface Delegate {
    void invoke(Function<Double, Double> f, double a, double b);
}


public class aboba {
    public static void sum(Function<Double, Double> f, double a, double b) {
        System.out.println("Sum = " + (f.apply(a) + b));
    }

    public static void multiply(Function<Double, Double> f, double a, double b) {
        System.out.println("Multiply = " + (f.apply(a) * b));
    }

    public static void main(String[] args) {
        Delegate del;

        del = aboba::sum;
        del.invoke(x -> x * x, 3, 4);

        del = aboba::multiply;
        del.invoke(x -> x + 2, 3, 4);
    }
}
