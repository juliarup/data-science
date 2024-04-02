import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf

def remove_outliers(df, column_names):
#wartości odstające
    for column in column_names:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] > upper_bound, Q3, df[column])
        df[column] = np.where(df[column] < lower_bound, Q1, df[column])
    return df

#ustawienia początkowe, które będą jednakowe dla obu sidebarów
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Projekt PAD</h1>", unsafe_allow_html=True)
st.title('Informacje o zbiorze danych diamentów')

st.write("""
Zbiór danych zawiera informacje na temat diamentów, z następującymi kolumnami:

- **carat**: Waga diamentu.
- **clarity**: Klasyfikacja czystości diamentu.
- **color**: Kolor diamentu.
- **cut**: Jakość szlifu diamentu.
- **x_dimension**: Długość w mm.
- **y_dimension**: Szerokość w mm.
- **z_dimension**: Głębokość w mm.
- **depth**: Głębokość diamentu.
- **table**: Szerokość wierzchu diamentu w stosunku do najszerszego punktu.
- **price**: Cena diamentu.
""")

df = pd.read_csv('cleaned_data.csv')
st.header('Dane')
st.dataframe(df)

###########################################################################################
# Sidebar
sidebar_options = ['Wizualizacja rozkładu zmiennych', 'Model regresji']
selected_option = st.sidebar.selectbox('Wybierz opcję:', sidebar_options)

# 1 sidebar
if selected_option == 'Wizualizacja rozkładu zmiennych':
    st.header('Korelacja między zmiennymi')
    st.write("""
- Karat wykazuje silną korelację z wymiarami diamentu (x, y, z) - większe diamenty będą ważyć więcej.
- Cena ma korelację z masą (karat) i wymiarami diamentu - większe diamenty zazwyczaj kosztują więcej.
""")
 ###########################################################################################   
    # Mapa ciepła
    corr_matrix = df.corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                            labels=dict(color="Współczynnik korelacji"),
                            x=corr_matrix.columns, y=corr_matrix.columns)
    fig_heatmap.update_layout(width=800, height=800, title_text='Mapa ciepła korelacji między zmiennymi')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
###########################################################################################    
# Wykres pudełkowy
    st.header('Wykres pudełkowy dla wybranej zmiennej')

    st.write("""
- Wykres pudełkowy jest przydatny w interpretacji wartości odstających.
- Wartości odstające widać w atrybutach: y_dimension, z_dimension, depth, price.
- Zobacz jak wartości odstające wpływają na wygląd wykresów.
""")
    if st.checkbox('Usuń wartości odstające przed analizą?'):
        df = remove_outliers(df, ['y_dimension', 'z_dimension', 'depth', 'price'])
        st.success('Wartości odstające zostały usunięte.')
                
    all_columns_option = 'Wszystkie'
    selected_variable = st.selectbox('Wybierz zmienną:', [all_columns_option] + list(df.columns))

    if selected_variable == all_columns_option:
        # Wyświetlenie wszystkich wykresów pudełkowych na raz
        cols = 5
        rows = 2
        fig_box_all = make_subplots(rows=rows, cols=cols, subplot_titles=df.columns)
        for i, column in enumerate(df.columns):
            box = go.Box(y=df[column], name=column)
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig_box_all.append_trace(box, row, col)
        fig_box_all.update_layout(height=600, width=600, title_text="Wykres pudełkowy dla wszystkich atrybutów", showlegend=False)
        st.plotly_chart(fig_box_all, use_container_width=True)
    else:
        # Wyświetlenie wykresu pudełkowego dla wybranej zmiennej
        fig_box = px.box(df, y=selected_variable)
        fig_box.update_layout(height=800, width=600, title=f"Wykres pudełkowy dla zmiennej:  '{selected_variable}'")
        st.plotly_chart(fig_box, use_container_width=True)
    
###########################################################################################
# Wykres Zależności ceny od poszczególnych atrybutów
#scatter
   
    st.header('Wykres zależności ceny od poszczególnych atrybutów')

    if st.checkbox('Usuń wartości odstające przed analizą ?'):
        df = remove_outliers(df, ['y_dimension', 'z_dimension', 'depth', 'price'])
        st.success('Wartości odstające zostały usunięte.')
                
    all_columns_option = 'Wszystkie'
    selected_variable = st.selectbox('Wybierz zmienną  :', [all_columns_option] + list(df.columns))

    if selected_variable == all_columns_option:
        # Wszystkich
        cols = 5
        rows = 2
        fig_scc = make_subplots(rows=rows, cols=cols, subplot_titles=[f'price({col})' for col in df.columns])
        for i, col in enumerate(df.columns):
            row = (i // cols) + 1
            col_index = (i % cols) + 1
            fig_scc.add_trace(go.Scatter(x=df[col], y=df['price'], mode='markers', name=col), row=row, col=col_index)
        fig_scc.update_layout(height=600, width=600, title_text="Zależności ceny od poszczególnych atrybutów", showlegend=False)
        st.plotly_chart(fig_scc, use_container_width=True)
    else:
        # dla wybranej zmiennej
        fig_scc = px.scatter(df, x=selected_variable, y='price', labels={'y': 'Price', 'x': selected_variable})
        fig_scc.update_layout(height=800, width=600, title=f"Zależność ceny od {selected_variable}")
        st.plotly_chart(fig_scc, use_container_width=True)

###########################################################################################
# histogram- liczebnosc

    st.header('Liczebność kategorii dla poszczególnych atrybutów')
    if st.checkbox('Usuń wartości odstające przed analizą??'):
        df = remove_outliers(df, ['y_dimension', 'z_dimension', 'depth', 'price'])
        st.success('Wartości odstające zostały usunięte.')
        
    # Opcja wyboru
    all_columns_option2 = 'Wszystkie'
    selected_variable2 = st.selectbox('Wybierz zmienną :', [all_columns_option2] + list(df.columns))

    if selected_variable2 == all_columns_option2:
        # wszystkie
        cols = 5
        rows = 2

        fig_histogram_all = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{col}' for col in df.columns])

        # Dodawanie histogramów
        for i, column in enumerate(df.columns):
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig_histogram_all.add_trace(go.Histogram(x=df[column], name=column), row=row, col=col)

        fig_histogram_all.update_layout(height=600, width=600, showlegend=False)
        st.plotly_chart(fig_histogram_all, use_container_width=True)
    else:
        # dla wybranej zmiennej
        fig_histogram = go.Figure(data=[go.Histogram(x=df[selected_variable2])])
        fig_histogram.update_layout(height=800, width=600, title=f"Histogram of {selected_variable2}")
        st.plotly_chart(fig_histogram, use_container_width=True) 
 
###########################################################################################
###########################################################################################
# model regresji
elif selected_option == 'Model regresji':
    st.header('Model regresji')

    # wartości odstające
    if st.checkbox('Usuń wartości odstające przed analizą?'):
        df = remove_outliers(df, ['y_dimension', 'z_dimension', 'depth', 'price'])
        st.success('Wartości odstające zostały usunięte.')

    # Wybór wielu zmiennych do modelu
    selected_features = st.multiselect('Wybierz zmienne, które najlepiej przewidują cenę:', df.columns, default=df.columns[0])
    formula = 'price ~ ' + ' + '.join(selected_features)

    # podsumowania modelu regresji
    model = smf.ols(formula=formula, data=df).fit()
    with st.expander("Zobacz podsumowanie modelu"):
        st.write(model.summary())

    # Wizualizacja tego co wybrał uzytkownik
    df['fitted'] = model.fittedvalues
    fig_regression = px.scatter(df, x='fitted', y='price', trendline='ols')
    fig_regression.update_layout(title="Model regresji", xaxis_title="Przewidywane ceny", yaxis_title="Rzeczywiste ceny", legend_title="Legenda")
    fig_regression.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
    st.plotly_chart(fig_regression, use_container_width=True)

    if st.button('Pokaż najlepsze dopasowanie modelu'):

        df = remove_outliers(df, ['y_dimension', 'z_dimension', 'depth', 'price'])
        vars = [col for col in df.columns if col != 'price']

        best_adj_r_squared = -float("inf")
        best_formula = ""

        model_list = []  # Przechowywanie modeli

        while vars:
            best_current_var = None
            for var in vars:
                formula = f"price ~ {best_formula} + {var}" if best_formula else f"price ~ {var}"
                model = smf.ols(formula=formula, data=df).fit()
                # Aktualizacja najlepszego modelu, jeśli obecny jest lepszy
                if model.rsquared_adj > best_adj_r_squared:
                    best_adj_r_squared = model.rsquared_adj
                    best_current_var = var

            # aktualizujemy najlepszą formułę i kontynuujemy
            if best_current_var:
                best_formula = f"{best_formula} + {best_current_var}" if best_formula else best_current_var
                vars.remove(best_current_var)
                model_list.append((best_formula, best_adj_r_squared))
            else:
                break

        # najlepszy model
        if best_formula:
            best_model_formula = f"price ~ {best_formula}"
            best_model = smf.ols(formula=best_model_formula, data=df).fit()
            st.write(f"""
                     - Najlepszy model: {best_model_formula}, 
                     - Adj. R-squared: {best_adj_r_squared:.4f}""")
            #st.write(best_model.summary())

            # Wizualizacja
            df["fitted"] = best_model.fittedvalues
            fig_regression = go.Figure()
            fig_regression.add_trace(go.Scatter(x=df["fitted"], y=df["price"], mode='markers', name='Dane'))
            fig_regression.add_trace(go.Scatter(x=df["fitted"], y=df["fitted"], mode='lines', name='Model regresji'))
            fig_regression.update_layout(title="Najlepszy model regresji", xaxis_title="Przewidywane", yaxis_title="Rzeczywiste")
            st.plotly_chart(fig_regression, use_container_width=True)
        else:
            st.write("Nie udało się znaleźć najlepszego modelu.")