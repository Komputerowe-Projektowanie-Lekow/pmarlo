# 1

## Ogląd trzech eksperymentów

Krótko: **eksperyment 1 daje sygnał metodologiczny** (pipeline działa, hipoteza nie do testu), **eksperyment 2 nie testuje niczego** bo bias jest 5-10x za słaby, **eksperyment 3 walidnuje implementację jakościowo** ale na warunkach, w których oba narzędzia są pod-progowe.

## Eksperyment 1: mdshare replay

To, co miało się stać, się stało. Grid kompletny: 36 warunków × 10 replikat = 360 wierszy, wszystko policzone. KL między warunkami waha się 0.33 do 0.89. Najlepsza strategia: `Fixed-T | Reweighted-Window | Fixed-50ep` z KL ≈ 0.33.

Problem: stosunek wariancji między-warunkami do wariancji wewnątrz-warunku wynosi **0.14**. To znaczy, że szum próbkowania w obrębie jednego warunku jest **siedem razy większy** niż różnice między warunkami. Ranking strategii w wykresie "Best KL strategies" jest praktycznie nieodróżnialny od permutacji losowej. Dokładnie tak miało być, bo strumień jest stacjonarny i strategie adaptacji nie mają czego adaptować.

Wniosek: pipeline działa numerycznie, reweighting nie wybucha, KL liczy się stabilnie. Sanity-check zaliczony. Nic więcej z tej tabeli nie powinno wychodzić do prezentacji jako "ranking strategii".

## Eksperyment 2: Müller-Brown z aktywnym biasem

Tu jest poważny problem. Trzy warstwy.

**Warstwa pierwsza, najgorsza: zero przejść.** Wszystkie 16 replik mają `transition_count = 0`, `unique_transition_count = 0`, `first_A_to_B = NaN`, `first_A_to_C = NaN`. Coverage w przestrzeni (x, y) waha się 2.5% do 4.8% siatki. Trajektorie spędziły cały czas w basenie A. Bez przejść nie ma sygnału jakości CV, nie ma sygnału adaptacji, nie ma sygnału hipotezy. Wszystkie wartości KL ~4.5-5.7 w wykresie "Best MB KL strategies" mierzą tylko jak bardzo trajektoria nie wyszła z basenu.

**Warstwa druga: niedoważony budżet biasu.** Z protokołu: `budget_frames = 6000`, `hill_stride = 500`, `hill_height = 0.5`. To daje 12 hillsów na run × 0.5 = 6.0 jednostek akumulowanej energii biasu, rozproszone po basenie. Bariera A→B w MB to około 38 jednostek energii. Czyli akumulowany bias to ~16% bariery, lokalnie pewnie ~10%. Plus `kT = 15`, więc `bariera/kT ≈ 2.5`, co oznacza, że bez biasu też nie byłoby spontanicznych przejść. Bias jest po prostu za słaby, żeby cokolwiek się stało.

**Warstwa trzecia: grid eksperymentu jest okrojony.** Zamiast planowanych 3 × 4 × 3 = 36 warunków z pasted2, w protokole jest 2 × 2 × 2 = 8 warunków. Brakuje:
- triggera `Threshold-delta` (jest tylko `Fixed-T` i `ADWIN`)
- polityki danych `Full` i `Reweighted-Full` (jest tylko `Window-W` i `Reweighted-Window`)
- polityki treningu `Fixed-200ep` (jest `Fixed-50ep` z max_epochs=10 i `EarlyStopping` z max_epochs=20)

Repliki: 2 zamiast 10. Plus `max_epochs=10` dla "Fixed-50ep" to mylące nazewnictwo, bo nazwa sugeruje 50 epok.

Wniosek: nawet gdyby bias działał, ten grid nie pokrywa wszystkich planowanych warunków. A na razie nie działa nic, bo bias jest pod-progowy.

## Eksperyment 3: walidacja PLUMED pesmd vs NumPy

Tu jest dobra wiadomość metodologiczna i zła wiadomość fizyczna.

**Dobra wiadomość.** PLUMED pesmd dostarczył 50 hillsów na 25000 kroków, NumPy MetaD ten sam protokół z 50 hillsów na 2500 frames raportowanych. Obydwie trajektorie zostały na 99% w basenie A, obie próbkują podobny region (x ∈ [-1.1, -0.1], y ∈ [1.0, 1.8] dla PLUMED; x ∈ [-1.35, -0.16], y ∈ [0.45, 1.88] dla NumPy). Wizualnie na wykresie 3 wyglądają niemal identycznie. To jest sukces walidacyjny w sensie: twoja własna implementacja MetaD odtwarza PLUMED jakościowo.

**Zła wiadomość.** Walidacja odbyła się na warunkach, w których oba narzędzia są jednakowo nieskuteczne. KL względem analitycznej referencji wynosi 3.65 dla PLUMED i 3.51 dla NumPy. To są te same wartości w jednostkach jednego basenu, bo żadna trajektoria nie zna basenów B i C. L1-dystans między dwoma histogramami wynosi 0.60, co jest duże, ale to różnica między tym, gdzie dokładnie w basenie A pojawiały się próbki, a nie różnica fundamentalna.

Czyli: walidacja pokazuje, że dwie implementacje dają jakościowo to samo, ale obie są tak ograniczone, że nie można powiedzieć, czy zgadzałyby się też w reżimie z faktycznymi przejściami.

## Co zrobić, żeby drugi eksperyment zaczął testować hipotezę

Trzy konkretne zmiany, w kolejności ważności.

**Zmiana 1, najważniejsza: zwiększyć budżet biasu o rząd wielkości.** Trzy opcje równoważne, można wybrać:

```
option_A: HILL_HEIGHT = 2.0,  BUDGET_FRAMES = 6000,  HILL_STRIDE = 500
          -> 12 hillsów * 2.0 = 24 jednostki, ~63% bariery (brzeg granicy)
option_B: HILL_HEIGHT = 1.0,  BUDGET_FRAMES = 25000, HILL_STRIDE = 500
          -> 50 hillsów * 1.0 = 50 jednostek, ~130% bariery
option_C: HILL_HEIGHT = 0.5,  BUDGET_FRAMES = 25000, HILL_STRIDE = 250
          -> 100 hillsów * 0.5 = 50 jednostek, ~130% bariery, ale gęstsze deponowanie
```

Polecam option_B. Powód: 25000 frames jest takie samo jak budżet sekcji 7, więc utrzymujesz porównywalność między dwoma eksperymentami. Z HEIGHT=1.0 i barierą 38 widzisz pierwsze przejścia A→B prawdopodobnie po 20-30 hillsach, czyli w okolicach kroków 10000-15000. Drugie przejście A→C wymaga dalszej akumulacji, ale w 25000 frames powinno się załapać przynajmniej jedno przejście w większości replikat.

**Zmiana 2: uzupełnić grid.** Dodać brakujące:
- `Threshold-delta` jako trigger
- `Full` i `Reweighted-Full` jako polityki danych
- `Fixed-200ep` jako polityka treningu (z prawdziwie 200 epokami, nie 10)

I podnieść `REPLICATES_PER_CONDITION` z 2 do 10. To 36 warunków × 10 = 360 replik. Przy proponowanym `BUDGET_FRAMES = 25000` i koszcie integratora Langevina w NumPy rzędu kilku sekund per replika, całość zmieści się w 30-60 minut na CPU w Colab.

**Zmiana 3, opcjonalna: obniżyć `kT` lub podgrzać do well-tempered.** Obecnie `kT=15` i bariera ~38 dają `bariera/kT = 2.5`, co jest na granicy reżimu, gdzie metadynamika jest tanio skuteczna. Dwie opcje:
- Obniżyć `kT` do 10 (bariera/kT = 3.8) - sprawia, że spontaniczne przejścia są jeszcze rzadsze, więc sygnał z metadynamiki staje się czystszy.
- Włączyć well-tempered z `bias_factor = 10` (już jest w protokole jako parametr, ale nie wiem czy jest używany w runnerze) - daje gwarantowaną zbieżność do pseudo-FES i pozwala porównywać KL między warunkami bezpośrednio.

## Podsumowanie statusu

Eksperyment 1 jest kompletny i daje wniosek negatywny w pożytecznym sensie: pipeline działa, hipoteza nie jest testowana, ranking jest szumem. To było zaprojektowane.

Eksperyment 2 wymaga restartu z poprawionym budżetem biasu. W obecnej formie nie da się z niego nic wyciągnąć o hipotezie, bo wszystkie warunki są równo zdegenerowane (zero przejść).

Eksperyment 3 spełnił swoją rolę walidacyjną jakościowo. Po naprawieniu eksperymentu 2 warto powtórzyć walidację na nowych parametrach (HEIGHT=1.0, 25000 frames), żeby zobaczyć, czy PLUMED i NumPy zgadzają się też w reżimie z przejściami. Wtedy walidacja będzie miała wagę.

Pytanie organizacyjne, którego nie umiem rozstrzygnąć z samych danych: czy `on_retrain_policy = "reproject_centers"` w eksperymencie 2 oznacza, że stare hillsy są przeliczane na nowe CV po retreningu, czy że nie są wcale używane. To istotna decyzja projektowa i mogłaby wpływać na sygnał, ale obecnie i tak nie ma sygnału, więc to do rozstrzygnięcia po pierwszej zmianie.