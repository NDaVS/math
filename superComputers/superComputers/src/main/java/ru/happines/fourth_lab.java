package ru.happines;

import java.util.concurrent.CompletableFuture;

public class fourth_lab {
    public static int countLetters(String str, char ch) {
        int i = 0;
        for (char c : str.toCharArray()){
            if (c == ch) i++;
        }

        return i;
    }

    public static void main(String[] args) {
    String input = "Вели медведя на ратное поле. Нет не на цепи - по собственной воле. За чужую свободу рвать убивать.";
    char target_char = 'е';

        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> countLetters(input, target_char));

        future.whenComplete((result, exception) -> {
            if (exception != null) {
                System.out.println("Ошибка при работе: " + exception.getMessage());
            } else {
                System.out.println("Количество вхождений символа: " + result);
            }
        });

        try {
            Thread.sleep(1000);
        }catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
