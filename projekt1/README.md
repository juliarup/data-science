# Projekt Analizy Danych: Diamenty

## Opis Projektu
Projekt skupia się na analizie, przetwarzaniu i wizualizacji danych dotyczących diamentów. Celem jest zrozumienie wpływu różnych cech diamentów na ich cenę oraz prezentacja wyników analizy za pomocą interaktywnego dashboardu.

## Struktura Projektu
Projekt składa się z kilku głównych części:
- `1_czyszczenie_danych.ipynb`: Proces wstępnej analizy i czyszczenia danych. Obejmuje usuwanie duplikatów, wartości odstających, normalizację nazw atrybutów, oraz ujednolicenie wartości.
- `2_wizualizacja_model.ipynb`: Wizualizacje danych i budowa modelu regresji. Zawiera heatmapy, wykresy pudełkowe, oraz analizę zależności między różnymi cechami a ceną diamentów.
- `3_dashboard_streamlit.py`: Skrypt Pythona do tworzenia interaktywnego dashboardu z wykorzystaniem Streamlit, prezentującego wyniki analizy.
- `messy_data.csv`: Plik źródłowy zawierający informacje o diamentach, wymagający przetworzenia.
- `cleaned_data.csv`: Zbiór danych po procesie czyszczenia, gotowy do analizy i wizualizacji.

## Technologie
W projekcie wykorzystano Python, Jupyter Notebook, bibliotekę Streamlit oraz różne biblioteki do analizy danych (pandas, numpy) i wizualizacji (matplotlib, seaborn).
