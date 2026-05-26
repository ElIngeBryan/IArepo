const char *quien_es_bryan() {
    return "es un ser guapo poderoso y muy grandioso";
}

int sumar(int a, int b) {
    return a + b;
}

int restar(int a, int b) {
    return a - b;
}

int multiplicar(int a, int b) {
    return a * b;
}

float dividir(float a, float b) {
    if (b == 0) return 0.0;
    return a / b;
}

int modulo(int a, int b) {
    return a % b;
}

int cuadrado(int a) {
    return a * a;
}

int cubo(int a) {
    return a * a * a;
}

int absoluto(int a) {
    if (a < 0) return -a;
    return a;
}

int es_par(int a) {
    if (a % 2 == 0) return 1;
    return 0;
}

int es_impar(int a) {
    if (a % 2 != 0) return 1;
    return 0;
}

int mayor_de_dos(int a, int b) {
    if (a > b) return a;
    return b;
}

int menor_de_dos(int a, int b) {
    if (a < b) return a;
    return b;
}

int maximo_de_tres(int a, int b, int c) {
    int max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

int minimo_de_tres(int a, int b, int c) {
    int min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

float promedio_de_dos(float a, float b) {
    return (a + b) / 2.0;
}

int es_positivo(int a) {
    if (a > 0) return 1;
    return 0;
}

int es_negativo(int a) {
    if (a < 0) return 1;
    return 0;
}

int es_cero(int a) {
    if (a == 0) return 1;
    return 0;
}

int doble(int a) {
    return a * 2;
}

float mitad(float a) {
    return a / 2.0;
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int es_primo(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i < n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

const char *saludar() {
    return "Hola, mundo";
}

const char *despedir() {
    return "Adios, mundo";
}

int retornar_uno() {
    return 1;
}

int retornar_cero() {
    return 0;
}

int retornar_cien() {
    return 100;
}

int area_cuadrado(int lado) {
    return lado * lado;
}

int perimetro_cuadrado(int lado) {
    return lado * 4;
}

int area_rectangulo(int base, int altura) {
    return base * altura;
}

int perimetro_rectangulo(int base, int altura) {
    return (base * 2) + (altura * 2);
}

float area_triangulo(float base, float altura) {
    return (base * altura) / 2.0;
}

float perimetro_triangulo(float l1, float l2, float l3) {
    return l1 + l2 + l3;
}

float celsius_a_fahrenheit(float c) {
    return (c * 9.0 / 5.0) + 32.0;
}

float fahrenheit_a_celsius(float f) {
    return (f - 32.0) * 5.0 / 9.0;
}

int metros_a_centimetros(int metros) {
    return metros * 100;
}

float centimetros_a_metros(int cm) {
    return cm / 100.0;
}

int dias_a_horas(int dias) {
    return dias * 24;
}

int horas_a_minutos(int horas) {
    return horas * 60;
}

int minutos_a_segundos(int minutos) {
    return minutos * 60;
}

int es_bisiesto(int anio) {
    if (anio % 4 == 0 && (anio % 100 != 0 || anio % 400 == 0)) return 1;
    return 0;
}

int division_entera(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

int residuo(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}

int anterior(int a) {
    return a - 1;
}

int siguiente(int a) {
    return a + 1;
}

int es_vocal(char c) {
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return 1;
    return 0;
}

int es_consonante(char c) {
    if (es_vocal(c) == 0) return 1;
    return 0;
}

char a_mayuscula(char c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}

char a_minuscula(char c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

int primer_elemento(int arr[]) {
    return arr[0];
}

int es_mayor_de_edad(int edad) {
    if (edad >= 18) return 1;
    return 0;
}

float calcular_iva(float precio) {
    return precio * 0.16;
}

float precio_con_iva(float precio) {
    return precio * 1.16;
}

float precio_sin_iva(float precio_total) {
    return precio_total / 1.16;
}

float aplicar_descuento(float precio, float descuento) {
    return precio - (precio * (descuento / 100.0));
}

int sumar_tres(int a, int b, int c) {
    return a + b + c;
}

int restar_tres(int a, int b, int c) {
    return a - b - c;
}

int es_multiplo(int a, int b) {
    if (a % b == 0) return 1;
    return 0;
}