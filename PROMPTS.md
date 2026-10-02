# PROMPTS

data|godzina|ID rozmowy|prompt|odpowiedź|
|:----|-------|----------|------|---------|
||||||

Spójrz na plik [ann\\_cls.ipynb](scripts/introduction/ann_cls.ipynb) i wykonaj w następującej kolejności nw. czynności:
1. Oceń implementację treningu oraz ewaluacji i wskaż polecenia, które są niepoprawne lub są potencjalnie problematyczne
2. Zaproponuj dla każdego z poleceń odrębne rozwiązanie, które zachowa styl programistyczny
3. Wygeneruj tutaj osobne okienko skryptowe, które implementuje mini-batch training oraz pętlę treningową wykorzystującą zarówno mini-batch training, jak i full-batch training na odrębnych kopiach modelu

Czas przetwarzania: 1m 28s
Model: GPT 5.5
Przetwarzanie: Wysokie

Odpowiedź:

Poniżej odnoszę się do [ann_cls.ipynb]. Najważniejszy problem: `Company_Tier` ma 4 klasy (`No Company`, `Tier 1`, `Tier 2`, `Tier 3`), a implementacja sieci neuronowej trenuje ją jak klasyfikator binarny.

**Problematyczne Polecenia**
| Polecenie | Problem | Proponowane rozwiązanie |
|---|---|---|
| `y_train_t = torch.tensor(train_y_t.reshape(-1, 1), dtype=torch.float32)` | Dla klasyfikacji wieloklasowej etykiety powinny być `LongTensor` i mieć kształt `[N]`, nie `[N, 1]`. | `y_train_cls_t = torch.tensor(train_y_t, dtype=torch.long)` |
| `nn.Linear(hidden_layer, 4)` | Liczba klas jest zahardkodowana. Działa tylko przypadkiem dla aktualnego targetu. | `nn.Linear(hidden_layer, len(le.classes_))` |
| `x = self.net(x).argmax(...).float()` w `forward()` | `argmax` w modelu odcina gradient. Model nie zwraca logitów, więc trening jest matematycznie błędny. | `return self.net(x)`; `argmax` wykonywać dopiero w ewaluacji. |
| `nn.BCEWithLogitsLoss(pos_weight=pos_weight)` | To loss binarny / multilabel, nie klasyfikacja jednej z 4 klas. | `nn.CrossEntropyLoss(weight=class_weights)` |
| `pos_weight = torch.tensor([counts[0] / counts[1]])` | Obsługuje tylko 2 klasy i zależy od kolejności `value_counts()`. | `torch.bincount(y_train_t)` i wagi dla wszystkich klas. |
| `torch.sigmoid(outputs) >= 0.5` | Próg 0.5 pasuje do binarnego wyjścia, nie do 4-klasowego softmax/logits. | `preds = logits.argmax(dim=1)` |
| `roc_curve(y_test, y_score)` | `roc_curve` jest binarne; przy 4 klasach wymaga OvR albo należy pominąć ROC. | `roc_auc_score(y_true, probs, multi_class="ovr", average="macro")` |
| `output_path = f"{OUTPUTS_DIR}/{datetime.now().strftime("%H-%M-%S")}"` | Niepoprawne cudzysłowy w f-stringu. | `output_path = f'{OUTPUTS_DIR}/{datetime.now().strftime("%H-%M-%S")}'` |
| `print(model(...).argmax(...))` | Po poprawieniu modelu ma sens, ale teraz jest podwójnie mylące, bo model już robi `argmax`. | `logits = model(...); print(logits.argmax(dim=1), y_test_cls_t[0])` |
| `study.optimize(...); return evaluate_model(... x_test_t ...)` | Optuna stroi hiperparametry na zbiorze testowym, czyli test przestaje być uczciwym testem. | Wydzielić `val_x/val_y` z treningu i test zostawić wyłącznie na końcową ocenę. |
| `BernoulliNB` na `OneHotEncoder + StandardScaler` | `BernoulliNB` zakłada cechy binarne; wystandaryzowane numeryczne cechy mogą być ujemne. | Dla NB użyć osobnego preprocessingu binarnego albo innego modelu, np. `GaussianNB` po konwersji do dense. |

**Okienko Skryptowe**
```python
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

torch.manual_seed(RANDOM_STATE)

y_train_cls_t = torch.tensor(train_y_t, dtype=torch.long)
y_test_cls_t = torch.tensor(test_y_t, dtype=torch.long)

class DeepNet(nn.Module):
    def __init__(self, input: int = 26, hidden_layer: int = 26 * 4, output: int = 4, dropout=0.1):
        super(DeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input, hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer, output)
        )

    def forward(self, x):
        return self.net(x)


def get_class_weights(y_t):
    counts = torch.bincount(y_t)
    weights = counts.sum() / (len(counts) * counts.float().clamp_min(1))
    return weights


def train_model_full_batch(model, x_train_t, y_train_t, epochs=200, lr=0.001,
                           weight_decay=1e-5, logging_step=10, l1_param=0,
                           output_path=f"{OUTPUTS_DIR}/full-batch"):
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(y_train_t))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    accuracies = []
    all_epochs = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        total_loss = loss + sum(param.abs().sum() for param in model.parameters()) * l1_param

        total_loss.backward()
        optimizer.step()

        if logging_step != -1 and epoch % logging_step == logging_step - 1:
            preds = logits.argmax(dim=1)
            acc = (preds == y_train_t).float().mean().item()

            print(f"Full batch | Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

            losses.append(total_loss.item())
            accuracies.append(acc)
            all_epochs.append(epoch + 1)
            plot_training(losses, accuracies, all_epochs, output_path)

    return losses, accuracies


def train_model_mini_batch(model, x_train_t, y_train_t, epochs=200, batch_size=128,
                           lr=0.001, weight_decay=1e-5, logging_step=10,
                           l1_param=0, output_path=f"{OUTPUTS_DIR}/mini-batch"):
    dataset = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=get_class_weights(y_train_t))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    accuracies = []
    all_epochs = []

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss = loss + sum(param.abs().sum() for param in model.parameters()) * l1_param

            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        if logging_step != -1 and epoch % logging_step == logging_step - 1:
            avg_loss = epoch_loss / total
            acc = correct / total

            print(f"Mini batch | Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

            losses.append(avg_loss)
            accuracies.append(acc)
            all_epochs.append(epoch + 1)
            plot_training(losses, accuracies, all_epochs, output_path)

    return losses, accuracies


def evaluate_model_multiclass(model, x_test_t, y_test_t, le, output_dir, log=True):
    model.eval()

    with torch.no_grad():
        logits = model(x_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        y_true = y_test_t.cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

        y_true_labels = le.inverse_transform(y_true)
        y_pred_labels = le.inverse_transform(y_pred)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels, target_names=le.classes_)

        try:
            roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = np.nan

        if log:
            print(f"Accuracy: {acc:.4f}")
            print(report)
            print(f"Macro ROC AUC OvR: {roc_auc:.4f}")

            plot_confusion_matrix(y_true_labels, y_pred_labels, output_dir)

    return acc


base_model = DeepNet(
    input=x_train_t.shape[1],
    hidden_layer=128,
    output=len(le.classes_),
    dropout=0.2
)

full_batch_model = deepcopy(base_model)
mini_batch_model = deepcopy(base_model)

timestamp = datetime.now().strftime("%H-%M-%S")
full_batch_output_path = f"{OUTPUTS_DIR}/{timestamp}-full-batch"
mini_batch_output_path = f"{OUTPUTS_DIR}/{timestamp}-mini-batch"

train_model_full_batch(
    full_batch_model,
    x_train_t,
    y_train_cls_t,
    epochs=200,
    lr=0.001,
    weight_decay=1e-5,
    logging_step=5,
    l1_param=0,
    output_path=full_batch_output_path
)

train_model_mini_batch(
    mini_batch_model,
    x_train_t,
    y_train_cls_t,
    epochs=200,
    batch_size=128,
    lr=0.001,
    weight_decay=1e-5,
    logging_step=5,
    l1_param=0,
    output_path=mini_batch_output_path
)

print("Full-batch evaluation")
full_batch_acc = evaluate_model_multiclass(
    full_batch_model,
    x_test_t,
    y_test_cls_t,
    le,
    full_batch_output_path
)

print("Mini-batch evaluation")
mini_batch_acc = evaluate_model_multiclass(
    mini_batch_model,
    x_test_t,
    y_test_cls_t,
    le,
    mini_batch_output_path
)

print(f"Full-batch accuracy: {full_batch_acc:.4f}")
print(f"Mini-batch accuracy: {mini_batch_acc:.4f}")
```

Najbardziej krytyczna poprawka to usunięcie `argmax` z `forward()` i przejście z `BCEWithLogitsLoss` na `CrossEntropyLoss`. Bez tego sieć nie uczy się poprawnie jako klasyfikator 4-klasowy.



###############################################

Przeczytaj [ann.ipynb](scripts/introduction/ann.ipynb) i wykonaj kroki w następującej kolejności:
1. Wskaż słabe punkty treningu i ewaluacji modelu regresyjnego
2. Objaśnij, co należy poprawić i jak
3. Wygeneruj poprawki włącznie z zaimplementowanym data_loaderem, który będzie trenował model mini-batchami

##################### ODPOWIEDŹ #################

### Slabe punkty treningu i ewaluacji modelu regresyjnego

1. Model regresyjny byl uczony funkcja `CrossEntropyLoss`, a ewaluacja uzywala accuracy, ROC AUC i macierzy pomylek. To sa narzedzia klasyfikacyjne, wiec nie mierza jakosci predykcji ceny.
2. `preprocessor.fit_transform()` byl wykonywany osobno na zbiorze testowym. To powoduje niespojna przestrzen cech i ryzyko data leakage. Zbior testowy powinien przechodzic tylko przez `preprocessor.transform()` nauczony na treningu.
3. `price_per_sqft` jest cecha pochodna od ceny, czyli target leakage. Model widzi informacje wyliczona z wartosci, ktora ma przewidywac.
4. `df.drop(columns=[...])` bez przypisania nie zmienial ramki danych, wiec czesc planowanego czyszczenia nie dzialala.
5. Cena ma duza skale i duzy rozrzut. Siec neuronowa stabilniej uczy sie na przeskalowanym targetcie, a metryki nalezy liczyc po odwroceniu skalowania do oryginalnych jednostek.
6. Full-batch trening slabiej wykorzystuje stochastycznosc optymalizacji. Mini-batche z `DataLoader` daja stabilniejszy i bardziej typowy proces uczenia.
7. Zbior testowy nie powinien sterowac decyzjami treningowymi ani doborem hiperparametrow. Do tego sluzy walidacja, a test zostaje na koncowy pomiar generalizacji.

### Co poprawiamy

- Uzywamy jednego preprocessora: `fit_transform()` tylko dla train, `transform()` dla validation i test.
- Dzielimy dane na train/validation/test, zeby test byl uzyty dopiero w finalnej ewaluacji.
- Skaluje target przez `StandardScaler`, trenuje na wartosciach przeskalowanych i raportuje MAE, RMSE oraz R2 w dolarach po `inverse_transform()`.
- Zastepuje trening klasyfikacyjny regresyjnym `MSELoss`.
- Dodaje `create_data_loader()`, ktory tworzy `TensorDataset` i `DataLoader` do uczenia mini-batchami.
- Wykresy treningu pokazuja `train_loss` i `val_loss`; ewaluacja pokazuje predykcje kontra wartosci rzeczywiste oraz residua.
