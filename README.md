# Sztuczne Sieci Neuronowe

Release v0.2.0

## Spis treści

- [Sztuczne Sieci Neuronowe](#sztuczne-sieci-neuronowe)
  - [Spis treści](#spis-treści)
  - [1 Wprowadzenie](#1-wprowadzenie)
    - [1.1 Geneza](#11-geneza)
    - [1.2 Perceptron](#12-perceptron)
    - [1.2 Funkcje aktywacji](#12-funkcje-aktywacji)
      - [W przeszłości](#w-przeszłości)
      - [ReLU](#relu)
      - [Lista funkcji aktywacji](#lista-funkcji-aktywacji)
    - [1.3 Wielowarstwowy perceptron - sieć głęboka](#13-wielowarstwowy-perceptron---sieć-głęboka)
      - [Budowa](#budowa)
      - [Propagacja w przód](#propagacja-w-przód)
      - [Propagacja w tył (Backpropagation)](#propagacja-w-tył-backpropagation)
      - [Interpretacje sieci głębokich](#interpretacje-sieci-głębokich)
    - [1.4 Istota treningu ANN](#14-istota-treningu-ann)
    - [1.5 Zastosowania](#15-zastosowania)
  - [2 Architektury ANN](#2-architektury-ann)
    - [2.1 Sieci konwolucyjne (Convolutional Neural Network)](#21-sieci-konwolucyjne-convolutional-neural-network)
    - [2.2 Generatywne Sieci Adwersalne (Generative Adversal Network)](#22-generatywne-sieci-adwersalne-generative-adversal-network)
    - [2.3 Sieci Rekurencyjne (Recurrent Neural Network)](#23-sieci-rekurencyjne-recurrent-neural-network)
    - [2.4 Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory)](#24-długa-pamięć-krótkoterminowa-lstm-long-short-term-memory)
    - [2.5 GAT](#25-gat)
    - [2.6 Autoenkodery](#26-autoenkodery)
    - [2.7 Transformery](#27-transformery)
      - [2.7.1 Mechanizm atencji](#271-mechanizm-atencji)
      - [2.7.2 Feed forward](#272-feed-forward)
      - [2.7.3 Enkoder](#273-enkoder)
      - [2.7.4 Dekoder](#274-dekoder)
  - [3 Dodatki](#3-dodatki)
    - [3.1 Optimizery](#31-optimizery)
      - [Stochastyczna optymalizacja gradientowa (SGD)](#stochastyczna-optymalizacja-gradientowa-sgd)
      - [SGD z pędem](#sgd-z-pędem)
      - [RMS-prop](#rms-prop)
      - [Adagrad](#adagrad)
      - [AdaDelta](#adadelta)
      - [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)
      - [AdamW](#adamw)
    - [3.2 Batch czy mini batch? Czyli o dzieleniu danych treningowych](#32-batch-czy-mini-batch-czyli-o-dzieleniu-danych-treningowych)
      - [Metoda spadku gradientu (batch gradient descent)](#metoda-spadku-gradientu-batch-gradient-descent)
      - [Metoda stochastycznego spadku gradientu (SGD)](#metoda-stochastycznego-spadku-gradientu-sgd)
      - [Mini-batch gradient descent](#mini-batch-gradient-descent)
    - [3.3 Sposoby na ograniczenie overfittingu](#33-sposoby-na-ograniczenie-overfittingu)
      - [Hold-out](#hold-out)
      - [Walidacja krzyżowa](#walidacja-krzyżowa)
      - [Regularyzacja](#regularyzacja)
      - [Dropout](#dropout)
      - [Eliminacja zmiennych nieistotnych](#eliminacja-zmiennych-nieistotnych)
      - [Wzbogacanie danych treningowych (data augmentation)](#wzbogacanie-danych-treningowych-data-augmentation)
      - [Ograniczanie złożoności modelu](#ograniczanie-złożoności-modelu)
    - [3.4 XAI](#34-xai)
    - [3.5 Metody treningu z niewielką lub żadną ilością danych](#35-metody-treningu-z-niewielką-lub-żadną-ilością-danych)
    - [3.6 Architektury modeli STT i TTS](#36-architektury-modeli-stt-i-tts)
  - [4 Bibliografia](#4-bibliografia)
  - [5 Słownik pojęć technicznych i anglojęzycznych](#5-słownik-pojęć-technicznych-i-anglojęzycznych)
    - [A - K](#a---k)
    - [J - Z](#j---z)
  
---

## 1 Wprowadzenie

Rozdział ten opowiada o istocie i zasadzie działania sztucznych sieci neuronowych. Opisuje ich genezę, budowę, zastosowania we współczesnym świecie i mechanizmy, które zachodzą zarówno podczas trenowania, jak i ewaluacji modeli opartych o sztuczne sieci neuronowe.

Pytania, na które poznasz odpowiedź w tym rozdziale.

- Jak zbudowane są głębokie sieci neuronowe?
- Na czym polega trening sieci neuronowej?
- Na czym polega trudność w wytrenowaniu sieci neuronowej?
- Jak można interpretować budowę głębokich sieci?

### 1.1 Geneza

Bezpośrednią inspiracją dla powstania sztucznych sieci neuronowych (które będę skrótowo odtąd nazywać ANN - Artificial Neural Network) jest budowa neuronów w ludzkim mózgu.

![image](imgs/neuron.png)

Neurony w mózgu składają się z dendrytów, jądra komórkowego, ciała komórkowego, aksonu i synaps. Dendrydy otrzymują sygnały z sąsiednich neuronów i przekazują je do ciała i jądra komórkowego, które modyfikują sygnał. Akson przekazuje nowy sygnał do synaps podłączonych do dendrydów innych neuronów.

Sygnały w mózgu przechodzą między neuronami, w których poddawane są indywidualnym procesom transformacji. Siła sygnału wyjściowego neuronu zależy od siły sygnałów wejściowych.

### 1.2 Perceptron

Twórcy koncepcji ANN zaproponowali, aby siła sygnałów była reprezentowana przez liczby rzeczywiste, a procesy transformacji polegały na obliczaniu wartości funkcji liniowej zawierającej tyle samo zmiennych, co wejść do danego sztucznego neuronu i zastosowaniu na nich wag, które pozwolą zbalansować wpływ różnych sygnałów na wielkość sygnału wyjściowego. Jest to bardzo ważne, gdyż bez tego pewne części ANN mogłyby w sposób niezamierzony (i na dodatek nieuczciwy) wpływać na wynik końcowy.

Przyjmijmy, że sieć neuronowa została wytrenowana do szacowania wartości mieszkania w zależności od metrażu, odległości od centrum i przeciętnych zarobków w tym mieście. Łatwo da się dostrzec, że dziedzina zmiennej opisującej przeciętne pensje mieści się w przedziale kilku, kilkunastu tysięcy. Gdyby nie stosować wag, to ta właśnie zmienna "przejęłaby kontrolę" nad modelem, co jest absolutnie niepożądane. Chcemy, aby każda zmienna w modelu miała wstępnie te same szanse.

Tak zbudowany neuron nazywamy perceptronem.

Jest jeszcze jedna rzecz, która odróżnia perceptron od zwyczajnych funkcji liniowych. Bez tej rzeczy sieci głębokie dałoby się uprościć do funkcji liniowych i nie miałyby żadnego zastosowania. Ten element odpowiada za nieliniowość w sieciach głębokich. Jest nim funkcja aktywacji.

### 1.2 Funkcje aktywacji

Funkcja aktywacji jest funkcją, która dla sumy wartości sygnałów i szumu dodawanego przez dany neuron zwraca nieliniowy sygnał na wyjście. Umożliwia ona odwzorowywanie nieliniowych zależności pomiędzy zmiennymi zależnymi (reprezentowanymi przez neurony warstwy końcowej), a zmiennymi wejściowymi (reprezentowanymi przez neurony w warstwie wejściowej). Dzięki temu ANN-y mogą uczyć się przewidywania nieliniowych zależności.

#### W przeszłości

W przeszłości używano funkcji trygonometrycznych takich jak funkcja sigmoidalna i tangens hiperboliczny (tanh). Niestety, badacze zauważyli, że powodują one kilka problemów.

1. Wykazują tendencję do nasycania się, co objawia się tym, że nieważne czy wejście ma dużą wartość, czy większą to zwraca ona bardzo małą pochodną, co straszliwie spowalnia trening;
2. Niewielkie wartości pochodnych dążące do 0 są piętą achillesową dla komputerów. Błędy numeryczne kumulują się wraz z obliczaniem kolejnych warstw, co utrudnia sprawne korygowanie wag.

#### ReLU

Aby rozwiązać oba te problemy zaproponowano funkcję ReLU (Rectified Linear Unit). Dla wartości niezerowych jest liniowa, ale dla ujemnych wartości zwraca zero. Dzięki temu nie nasyca się, a ponadto wykazuje się prostą pochodną, która eliminuje problem błędów numerycznych. Oczywiście są różne wariacje na temat funkcji ReLU, są jeszcze funkcje oparte o stałą $e$, ale na początek warto znać kilka podstawowych funkcji aktywacji.

#### Lista funkcji aktywacji

- Tanh

$$
f(x)=\frac{2}{1+e^{-2x}} - 1
$$

- Sigmoid

$$
f(x)=\frac{1}{1+e^{-x}}
$$

- ReLU*

$$
f(x)=max(0, x)
$$

- ELU

$$
f(x)=\begin{cases}
x, & x>0\\
\alpha (e^x-1), & x\le0
\end{cases}
$$

- Leaky ReLU

$$
f(x)=
\begin{cases}
x, & x>0\\
\alpha x, & x\le0
\end{cases}
$$

- SELU

$$
f(x)=\lambda \begin{cases}
x, & x>0\\
\alpha (e^x-1), & x\le0
\end{cases}
$$
$$
\lambda \approx 1.05
\alpha \approx 1.67
$$

- SoftPlus

$$
f(x)=log(1 + e^x)
$$

- Softmax*

$$
f(x_i)=\sigma(x_i)=\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

*- funkcje te są jedynymi dozwolonymi na warstwach wyjściowych

### 1.3 Wielowarstwowy perceptron - sieć głęboka

Sieci zbudowane są z wielu takich perceptronów ułożonych równolegle ze sobą tworząc warstwy sieci. Warstwy sieci z kolei są połączone szeregowo, co czyni je siecią głęboką. Najprostszą postacią sieci głębokiej jest perceptron wielowarstwowy, w skrócie MLP (Multi Layer Perceptron). W dalszej części kompendium pojawi się sieć sprzężenia do przodu (Feedforward Neural Network), która jest w zasadzie tym samym, z tym że nazwa nawiązuje do tego, jak model dokonuje obliczeń.

#### Budowa

Sieć głęboka (MLP) składają się kolejno z:

- warstwy wejściowej (input layer)
- warstw ukrytych (hidden layers)
- warstwy wyjściowej (output layer)

Warstwa wejściowa ma tyle neuronów, ile zmiennych jest wprowadzanych. Jeżeli któraś ze zmiennych jest kategorialna, należy zrzutować ją albo na liczby, albo na wektor zer i jedynek, co skutkuje oczywiście zwiększeniem liczby potrzebnych neuronów na początku.

Warstwy ukryte mogą mieć dowolną, niezerową liczbę neuronów. Zwykle pierwsza z nich ma więcej neuronów niż w warstwie wejściowej. O tym jak liczba neuronów może wpływać na zdolność sieci do prognozowania piszę w tym rozdziale o [tutaj](#interpretacje-sieci-głębokich).

Liczba neuronów w warstwie wyjściowej odpowiada liczbie klas szacowanej zmiennej zależnej. Jeżeli zmienna zależna jest ilościowa, to występuje tylko jeden neuron.

#### Propagacja w przód

Warstwa wejścia dostarcza danych liczbowych do neuronów pierwszej warstwy ukrytej. Każdy taki neuron z osobna w warstwie ma własny zestaw wag oraz parametr szumu, zwany *biasem*, którymi traktuje dane wejściowe. Suma iloczynu skalarnego wektora wag i wektora danych wejściowych oraz szumu po zastosowaniu funkcji aktywacji stanowi sygnał wyjściowy danego neuronu. Sygnał ten następnie jest przekazywany do następnej warstwy i ich neuronów i traktowany tak samo.

$$
n(x) = f(\sum_i{w_i x_i + b})
$$

Sygnały z ostatniej warstwy ukrytej dochodzą do warstwy wyjściowej.

Propagacja w przód polega na przekazywaniu sygnałów DO PRZODU warstwa po warstwie. Sygnał nie jest propagowany ani w kierunku tych samych neuronów w warstwie, ani do tyłu. W innych architekturach np. sieciach rekurencyjnych lub rezydualnych sygnał może być przekazywany od neuronu do neuronu w obrębie warstwy lub może je pomijać.

#### Propagacja w tył (Backpropagation)

Propagacja w tył jest algorytmem umożliwiającym wytrenowanie sztucznej sieci neuronowej. Proces treningu składa się z następujących kroków:

1. Ustaw wagi wstępne w modelu;
2. Użyj danych treningowych do przeprowadzenia propagacji w przód;
3. Wynik z propagacji porównaj z docelowym wynikiem i na jego podstawie oblicz błąd modelu;
4. Na podstawie wielkości błędu oblicz zmianę wag dla każdego neuronu warstwa po warstwie idąc wstecz;
5. Powtórz proces od kroku 2., jeżeli to była ostatnia iteracja lub błąd stał się akceptowalny.

A więc propagacja w tył to nic innego jak przerzucanie błędu modelu od warstwy końcowej na sam początek. Błąd można obliczyć korzystając z:

- różnicy;
- błędu średniokwadratowego;
- kwadratu odległości euklidesowej;
- entropii krzyżowej.

Ponieważ w warstwie wyjściowej znajduje się najczęściej więcej niż jeden neuron, a problemy są częściej z gatunku problemów klasyfikacyjnych, to posługujemy się raczej tą ostatnią metodą na obliczenie błędu. Odległość euklidesowa to po prostu błąd średniokwadratowy, lecz dla wektora parametrów wyjściowych. Używany jest do szacowania zmiennych ilościowych. Zaś entropia krzyżowa jest używana do zmiennych jakościowych (kategorialnych). Wyraża się ona wzorem:

$$
L(y,y')=-\sum_{i=1}^cy_ilog (y'_i)
$$

Gdzie

$y_i$ - Prawdziwa etykieta oznaczająca przynależność do klasy *i*

$y'_i$ - Przewidywane prawdopodobieństwo przynależności do klasy *i*

Aby można było oszacować zmianę wagi, skorzystamy z techniki gradientowej optymalizacji. Należy obliczyć gradient dla wyjścia modelu oraz przewidywanego wyjścia i odwrócić kierunek w stronę (lokalnego) optimum. Dla funkcji sigmoidalnej postaci:

$$
f(x)=\frac{1}{1+e^{-x}}
$$

pochodna funkcji to:

$$
\frac{d}{dx}f(x)=f(x)(1-f(x))=\frac{e^{-x}}{(1+e^{-x})^2}
$$

Dla funkcji ReLU:

$$
f(x)=max(0, x)
$$

pochodną funkcji jest:

$$
\frac{d}{dx}f(x)=\begin{cases}
1, & x>0 \\
0, & x<0
\end{cases}
$$

Oznaczenia:

$N$ - liczba warstw w sieci włącznie z warstwą wejściową i wyjściową \
$H$ - liczba warstw ukrytych

$M$ - liczba neuronów w warstwie ukrytej (przyjmijmy, że warstwy ukryte mają tyle samo neuronów)\
$I$ - liczba neuronów w warstwie wejścia \
$O$ - liczba neuronów w warstwie wyjścia

$w_{j,l}^{i,k}$, $i\lt j$ - waga pomiędzy $k$-tym neuronem $i$-tej warstwy, a $l$-tym neuronem $j$-tej warstwy\
$b_{i,j}$ - szum (bias) w $j$-tym neuronie $i$-tej warstwy\
$n_{i,j}$ - wartość sygnału neuronu w $j$-tym neuronie w $i$-tej warstwie

$\delta_{j,l}^{i,k}$, $i\lt j$ - zmiana wagi pomiędzy $k$-tym neuronem $i$-tej warstwy, a $l$-tym neuronem $j$-tej warstwy\
$e_{i,j}$ - błąd w $j$-tym neuronie $i$-tej warstwy\
$\lambda$ - współczynnik uczenia

Przyjmijmy, że:

$N=M=3$

$I=2$

$O=1$

Dla neuronu wyjściowego (warstwy wyjściowej) wzór na zmianę wag połączeń kończących się w nim jest następujący:

$
\delta_{3,1}^{2,k}=e_{3,1} \cdot \frac{d}{dx}f(n_{3,1})
$

$
w_{3,1}^{2,k} = w_{3,1}^{2,k} + n_{3,1}^T \cdot \delta_{3,1}^{2,k} \cdot \lambda
$

$
b_{3,1} = b_{3,1} + \sum_{k=1}^M \delta_{3,1}^{2,k}\\
$

$k=\overline{1,M}$

Dla warstwy ukrytej zmiany wag oblicza się w następujący sposób:

$
\delta_{2,l}^{1,k} = w_{2,l}^{1,k} \cdot \frac{d}{dx} f(n_{1,l}) \\
$

$
w_{2,l}^{1,k} = w_{2,l}^{1,k} + n_{1,l}^T \cdot \delta_{2,l}^{1,k}
$

$
b_{1,l} = b_{1,l} + \sum_{k=1}^M \delta_{2,l}^{1,k}\\
$

$k,l=\overline{1,M}$

Współczynnik uczenia ustawia się po to, aby parametry miały szansę odnaleźć lepsze optimum lokalne. Bez tego model natychmiast wpadnie w najbliższe optimum lokalne, które najczęściej będzie ono niesatysfakcjonujące.

#### Interpretacje sieci głębokich

Pierwszą interpretacją, jaką proponuje R. Hurbans w [RHu] jest to, że każda kolejna warstwa ANN tworzy coraz bardziej korelujące dane, które wreszcie stają się w pełni skorelowane na warstwie wyjściowej.

Drugą interpretacją zaproponowaną w [Wel] jest to, że sieć neuronową odwzorowuje mapa regionów decyzyjnych oddzielonych granicami decyzyjnymi. Im więcej neuronów, tym więcej granic i obszarów, lecz im więcej warstw w sieci, tym więcej takich obszarów i tym mniejszy koszt obliczeniowy. Dobrze oddaje to wzór na maksymalną liczbę regionów:

$$

N = (\frac{D}{D_i} + 1)^{D_i (K-1)}(\frac{D^2+D+2}{2})

$$

Gdzie:

$D_i$ - liczba neuronów w warstwie wejściowej \
$D$ - liczba neuronów na warstwę\
$K$ - liczba warstw pośrednich

W tym ujęciu sieci trzywarstwowe zawierające tylko jedną warstwę ukrytą są ukazane jako nieefektywne, ponieważ mają one mniejszą elastyczność wyrażaną liczbą regionów. Ze wzoru można wywnioskować, że przyrost liczby regionów jest wielomianowy, natomiast ten sam przyrost wywołany zwiększeniem liczby warstw jest wykładniczy.

Ogólnie rzecz biorąc, ciężko jest stworzyć czytelną i zrozumiałą interpretację modelu dla każdego problemu. Każdy neuron w sieci wykonuje swoje zadanie, które trudno jest opisać jednoznacznie i precyzyjnie. Ale są metody, które pomagają zrozumieć to, jak dana sieć neuronowa dochodzi do rozwiązań. Nimi zajmuje się osobna dziedzina badań - XAI (Explainable AI), dzięki której zyskujemy coraz lepszy wgląd w proces rozumowania systemów opartych na modelach AI, na podstawie którego można oceniać bezpieczeństwo i słuszność w podejściu tych modeli.

### 1.4 Istota treningu ANN

Właściwe ustawienie wag w sieci głębokiej polega na tym, że dla danego zestawu danych treningowych sieć głęboka musi zwracać jak najmniejszy błąd na wyjściu. Algorytm propagacji wstecz działa na zasadzie spadku gradientowego. Niestety (dla architektów sieci głębokich) albo na szczęście (bowiem taka jest rzeczywistość) świat jest bardziej skomplikowany niż funkcja liniowa. W przestrzeni rozwiązań dopuszczalnych są rozwiązania, które zwracają zaledwie minima lokalne funkcji błędu, ale jest też co najmniej jedno rozwiązanie, które zwraca minimum globalne. Gradienty mogą zwracać wartości, które niekoniecznie kierują na minimum globalne, lecz na minimum lokalne, co bez dwóch zdań utrudnia trening. Zatem podczas treningu trzeba zwracać uwagę na to, aby kierunek optymalizacji był z jak największym prawdopodobieństwem zgodny z położeniem minimum globalnego, ewentualnie położeniem rozwiązania w granicach dopuszczalnego błędu względem minimum globalnego.

Można to uprawdopodobnić na wiele sposobów. Na przykład podczas inicjalizacji wag losuje się kilka lub więcej zestawów, a następnie dokonuje się selekcji takiego zestawu, który zwraca najmniejszy błąd.

Zamiast algorytmu stochastycznego spadku gradientowego stosuje się inne, które na różnych etapach treningu promują bardziej eksplorację, niż eksploatację przestrzeni rozwiązań i vice versa. Robią to poprzez modyfikację współczynnika $\lambda$, który odpowiada za wielkość kroku (wyżarzanie kosinusowe). Robią to poprzez szacowanie pędów (momentów) gradientów (rodzina algorytmów Adam).

Kolejną sprawą jest zapobieganie przesadnemu dopasowaniu modelu do danych treningowych. Objawia się to tym, że dla danych treningowych model bardzo trafnie przewiduje wyniki, zaś dla danych spoza tego zbioru model cechuje się gorszą dokładnością, która w skrajnych sytuacjach będzie mniej lub bardziej podobna do zgadywania. W terminologii, która bardzo wiele zawdzięcza światu anglosaskiemu, nazywa się to **overfittingiem**. O sposobach na zapobieganie mu [piszę tutaj](#32-batch-czy-mini-batch-czyli-o-dzieleniu-danych-treningowych).

### 1.5 Zastosowania

Sztuczne sieci neuronowe stosuje się m.in. do:

- szacowania przyszłych cen akcji na giełdzie;
- oceny ryzyka kredytowego;
- tłumaczenia tekstów na obce języki;
- wykrywania chorób ze zdjęć RTG;
- analizy sentymentu na podstawie wpisów w Internecie;
- generowania muzyki
- generowania filmów i obrazów
- wspomagania podejmowania decyzji w złożonych środowiskach operacyjnych

## 2 Architektury ANN

Pytania, na które poznasz odpowiedź w tym rozdziale.

- Jak sieci głębokie rozpoznają ludzi i przedmioty na zdjęciach?
- Jak działają współczesne LLM-y, które ułatwiają nam życie?
- Na czym polega proces generowania deepfake'ów?

### 2.1 Sieci konwolucyjne (Convolutional Neural Network)

### 2.2 Generatywne Sieci Adwersalne (Generative Adversal Network)

### 2.3 Sieci Rekurencyjne (Recurrent Neural Network)

### 2.4 Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory)

### 2.5 GAT

### 2.6 Autoenkodery

### 2.7 Transformery

#### 2.7.1 Mechanizm atencji

#### 2.7.2 Feed forward

#### 2.7.3 Enkoder

#### 2.7.4 Dekoder

## 3 Dodatki

Pytania, na które poznasz odpowiedź w tym rozdziale.

- Czym różnią się od siebie algorytmy modyfikujące wagi (optimizery)?
- Jak skutecznie zapobiegać overfittingowi?
- Jak komputer potrafi rozpoznać mowę człowieka?

### 3.1 Optimizery

#### Stochastyczna optymalizacja gradientowa (SGD)

#### SGD z pędem

#### RMS-prop

#### Adagrad

#### AdaDelta

#### Adam (Adaptive Moment Estimation)

#### AdamW

### 3.2 Batch czy mini batch? Czyli o dzieleniu danych treningowych

Istnieją różne podejścia w przekazywaniu danych podczas pętli treningowej. Można podawać cały komplet, w oparciu o który model ustawia swoje wagi. Można podawać je pojedynczo, albo partiami liczącymi po kilka przykładów. Poniżej opisuję ich wady oraz zalety, które warto znać.

#### Metoda spadku gradientu (batch gradient descent)

Polega ona na obliczaniu nowych wag po przeprocesowaniu całego zbioru treningowego. Pozwala ona uzyskiwać dokładne poprawki, które są uśrednione dla całego zbioru treningowego. Wyróżnia się dodatkowo stabilnością, przez którą funkcja straty jest malejąca, co widać na wykresie spadku straty. Wadą jest pamięciożerność oraz czasochłonność, gdyż model musi obliczać w każdej epoce predykcje dla wszystkich przykładów.

#### Metoda stochastycznego spadku gradientu (SGD)

Metoda stochastyczna polega na obliczaniu nowych wag przy użyciu tylko jednej obserwacji ze zbioru treningowego.

Dzięki temu pętla treningowa zużywa mniej pamięci, ponieważ trzeba przechowywać mniej predykcji w każdej iteracji. Ta metoda jest też o wiele szybsza. Pozwala ona unikać minimów lokalnych. Świetnie nadaje się zarówno do ogromnych zbiorów danych, jak i systemów chmurowych, a także do urządzeń brzegowych (IoT, telefony, tablety).

Metoda ta za sprawą swoich zalet ma następujące wady. Po pierwsze, trening jest niestabilny, gdyż wagi są gwałtownie zmieniane, a informacja z poprzednich iteracji może zostać zatracona. Po drugie, model podczas treningu wykazuje wysokie tendencje do oscylowania wokół docelowego minimum.

Aby ta metoda była skuteczna, należy przed każdą epoką losować kolejność podawania danych treningowych.

#### Mini-batch gradient descent

Rozwiązaniem kompromisowym jest dzielenie zbioru treningowego na porcje danych liczące po kilka lub więcej obserwacji. To podejście pozwala zarówno ustabilizować trening, jak i zachować zalety poprzedniej metody takie jak unikanie minimów lokalnych. Jest szeroko wykorzystywana do trenowania ogromnych modeli przy użyciu dużych zbiorów danych.

Jednakże, aby wykorzystać pełen potencjał tej metody, należy próbować różnych rozmiarów tych batch'y (porcji danych). Zaleca się, aby liczba ta była potęgą dwójki, aby móc optymalnie wykorzystywać zasoby obliczeniowe kart graficznych. Można zacząć próbować od wielkości 16 lub 32 obserwacji.

### 3.3 Sposoby na ograniczenie overfittingu

#### Hold-out

Hold-out polega na dzieleniu zbioru danych na podzbiór treningowy i testowy. W praktyce często wyznacza się też osobny zbiór walidacyjny, który pozwala na bieżąco oceniać postęp treningu po każdej epoce.

Dzięki temu podziałowi można w banalny sposób ocenić, czy model jest nadmiernie dopasowany, czy nie. Wystarczy spojrzeć na metryki dokładności i stwierdzić, czy dla zbioru testowego są one istotnie mniejsze, niż dla zbioru treningowego. Jeżeli tak, to model jest nadmiernie dopasowany. Jeżeli nie, to oznacza, że model nauczył się generalizacji.

#### Walidacja krzyżowa

w przypadku większych modeli może okazać się zbyt kosztowna.

#### Regularyzacja

- Regularyzacja L1 (lasso) i L2 (ridge);

#### Dropout

- Dropout;

#### Eliminacja zmiennych nieistotnych

#### Wzbogacanie danych treningowych (data augmentation)

wprowadzanie do zbioru danych treningowych artefaktów, które w praktycznym zastosowaniu mogłyby zaburzać pracę, ale podczas treningu uodporniają model na anomalie otrzymane na wejściu. Sprowadza się to do dodawania mniej lub bardziej regularnych szumów, zakrywania części obrazu i innych manipulacji;

#### Ograniczanie złożoności modelu

### 3.4 XAI

SHAP, LIME, wykresy PDP, ICE, testy ANOVA jedno i dwukierunkowe

### 3.5 Metody treningu z niewielką lub żadną ilością danych

### 3.6 Architektury modeli STT i TTS

CNN, RNN i Transformery

## 4 Bibliografia

1. [Rhu] R. Hurbans. Grokking Artificial Intelligence Algorithms. Manning Publications Co. Rok wydania 2020. ISBN: 978-16-172-9618-5
2. L. Tunstall, L. von Werra, T. Wolf. Przetwarzanie języka naturalnego z wykorzystaniem transformerów. Helion S.A. 2024. ISBN: 978-83-289-0711-9
3. D. Chuan-En-Lin. 8 Simple Techniques to Prevent Overfitting, [online]. Dostęp w Internecie: <https://medium.com/data-science/8-simple-techniques-to-prevent-overfitting-4d443da2ef7d>. [dostęp: 31.07.2026]
4. GeeksforGeeks. Activation Functions in Neural Network, [online]. Dostęp w Internecie: <https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/>. [dostęp: 31.07.2026]
5. GeeksforGeeks. Backpropagation in Neural Network, [online]. Dostęp w Internecie: <https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/>. [dostęp: 03.08.2026]
6. Musstafa. Optimizers in Deep Learning, [online]. Dostęp w Internecie: <https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0>. [dostęp: 03.08.2026]
7. GeeksforGeeks. Optimization Rule in Deep Neural Networks, [online]. Dostęp w Internecie: <https://www.geeksforgeeks.org/deep-learning/optimization-rule-in-deep-neural-networks/>. [dostęp: 03.08.2026]
8. D. Altinel. Development of Deep Learning Optimizers: Approaches, Concepts, and Update Rules. Istanbul Medeniyet University. 22.09.2025. Dostęp w Internecie: <https://arxiv.org/pdf/2509.18396>.
9. A. Zhang, Z. C. Lipton, M. Li, A. J. Smola. Dive into Deep Learning. Release 0.16.1. 19.01.2021
10. [Wel] Illustrated guide to AI. Volume I. The Welch Labs. 2025
11. A. W. Trask. Zrozumieć głębokie uczenie. Wydawnictwo PWN. Warszawa 2019. ISBN: 978-83-01-20782-3
12. L. Bhuva. Mini-Batch Gradient Descent: A Comprehensive Guide, [online]. Dostęp w Internecie: <https://medium.com/@lomashbhuva/mini-batch-gradient-descent-a-comprehensive-guide-ba27a6dc4863>. [dostęp: 31.08.2026]

## 5 Słownik pojęć technicznych i anglojęzycznych

### A - K

Batch - wsad, partia, porcja (zwykle danych treningowych);

Celność - opisuje bliskość wyniku do wskazanego celu;

Dokładność - patrz: Celność;

Ex ante - łac. "przed faktem". Dotyczy właściwości modelu ocenianych na podstawie danych treningowych;

Ex post - łac. "po fakcie". Dotyczy właściwości modelu ocenianych na podstawie danych testowych;

Generalizacja - zdolność modelu do poprawnego przewidywania wyników dla sytuacji, w których nie był trenowany;

Konwergencja - zbieżność, podobieństwo;

### J - Z

Precyzja - opisuje miarę rozrzutu wyników. Wysoka precyzja oznacza niewielki rozrzut i vice versa;
