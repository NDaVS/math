syms x;
%result = (x-1) * (x-2) * (x-3) * (x-4) * (x-5) * (x-6) * (x-7) * (x-8) * (x-9) * (x-10) * (x-11) * (x-12) * (x-13) * (x-14) * (x-15) * (x-16) * (x-17) * (x-18) * (x-19) * (x-20);
%disp(expand(result));
eq = x^20 - (210+10^(-7))*x^19 + 20615*x^18 - 1256850*x^17 + 53327946*x^16 - 1672280820*x^15 + 40171771630*x^14 - 756111184500*x^13 + 11310276995381*x^12 - 135585182899530*x^11 + 1307535010540395*x^10 - 10142299865511450*x^9 + 63030812099294896*x^8 - 311333643161390640*x^7 + 1206647803780373360*x^6 - 3599979517947607200*x^5 + 8037811822645051776*x^4 - 12870931245150988800*x^3 + 13803759753640704000*x^2 - 8752948036761600000*x + 2432902008176640000;
coefficients = sym2poly(eq); % Получение коэффициентов квадратного уравнения
disp(coefficients);
answers = roots(coefficients);
disp(answers);
plot(real(answers), imag(answers), 'o');
xlabel('Re(z)');
ylabel('Im(z)');
title('График комплексных чисел');
grid on;