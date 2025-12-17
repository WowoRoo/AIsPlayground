# Detektor Sentymentu Rozmówców

Program do analizy sentymentu rozmówców w języku polskim, wykorzystujący modele oparte na architekturze BART/BERT.

## Opis

Program analizuje rozmowy w języku polskim i określa sentyment (pozytywny, negatywny, neutralny) dla każdej wypowiedzi oraz podsumowuje ogólny sentyment każdego rozmówcy.

## Wymagania

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)

## Instalacja

1. Zainstaluj wymagane biblioteki:

```bash
pip install -r requirements.txt
```

## Użycie

### Podstawowe użycie z przykładową rozmową:

```bash
python sentiment_detector.py
```

### Analiza rozmowy z pliku:

```bash
python sentiment_detector.py przykladowa_rozmowa.txt
```

### Zapis wyników do pliku JSON:

```bash
python sentiment_detector.py przykladowa_rozmowa.txt --output wyniki.json
```

### Użycie innego modelu:

```bash
python sentiment_detector.py przykladowa_rozmowa.txt --model nazwa_modelu
```

## Format pliku z rozmową

Rozmowa powinna być w formacie tekstowym, gdzie każda linia reprezentuje wypowiedź:

```
Rozmówca1: Tekst wypowiedzi
Rozmówca2: Tekst wypowiedzi
Rozmówca1: Kolejna wypowiedź
```

Przykład:
```
Anna: Cześć! Jak się masz?
Jan: Witaj! Świetnie, dziękuję!
```

## Modele

Domyślnie program używa modelu `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`, który:
- Jest wielojęzyczny i dobrze działa z polskim
- Jest zoptymalizowany do analizy sentymentu
- Jest oparty na architekturze RoBERTa (podobnej do BART)

Alternatywne modele do rozważenia:
- `dkleczek/bert-base-polish-cased-v1` - polski model BERT
- `allegro/herbert-base-cased` - polski model Herbert

## Przykładowe wyniki

Program wyświetla:
- Szczegółową analizę każdej wiadomości z określeniem sentymentu
- Podsumowanie dla każdego rozmówcy z dominującym sentymentem
- Statystyki (pozytywne/negatywne/neutralne wypowiedzi)
- Poziomy pewności dla każdej analizy

## Struktura projektu

```
Modele/
├── sentiment_detector.py      # Główny program
├── requirements.txt            # Zależności
├── przykladowa_rozmowa.txt     # Przykładowa rozmowa
└── README.md                   # Ten plik
```

## Uwagi

- Przy pierwszym uruchomieniu model zostanie pobrany z Hugging Face (może to zająć kilka minut)
- Modele wymagają sporo pamięci RAM (zalecane minimum 4GB)
- Jeśli masz kartę graficzną NVIDIA z CUDA, program automatycznie ją wykorzysta
- Długie teksty są automatycznie przycinane do 512 tokenów

## Licencja

Program używa modeli z Hugging Face, które mogą mieć różne licencje. Sprawdź licencję wybranego modelu przed użyciem komercyjnym.

