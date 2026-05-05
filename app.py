
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Загрузка модели и колонок
try:
    model = joblib.load('realty_model.pkl')
    model_columns = joblib.load('model_columns.pkl')

    st.title("Сервис оценки недвижимости")

    # 2. Форма ввода
    with st.form("input_form"):
        area = st.number_input("Площадь (кв. футы)", value=1500)
        year = st.number_input("Год постройки", value=2000)
        rooms = st.number_input("Комнат", value=5)
        condition = st.slider("Состояние (1-10)", 1, 10, 5)

        # Выбор района
        districts = [c.replace('Neighborhood_', '') for c in model_columns if 'Neighborhood_' in c]
        selected_dist = st.selectbox("Район", sorted(districts) if districts else ["NA"])

        submit = st.form_submit_button("Рассчитать")

    # 3. Логика предсказания
    if submit:
        #создаем dataframe из нулей (решает проблему NaN)
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        #заполняем числовые поля
        input_df.at[0, 'Gr Liv Area'] = area
        input_df.at[0, 'Year Built'] = year
        input_df.at[0, 'TotRms AbvGrd'] = rooms
        input_df.at[0, 'Overall Cond'] = condition

        #заполняем район (ставим 1 в нужную колонку)
        dist_col = f'Neighborhood_{selected_dist}'
        if dist_col in input_df.columns:
            input_df.at[0, dist_col] = 1

        #финальная проверка на NaN перед моделью
        input_df = input_df.fillna(0)

        #предсказание
        price = model.predict(input_df)[0]
        st.success(f"### Прогнозная цена: ${max(0, price):,.2f}")

except Exception as e:
    st.error(f"Ошибка: {e}")
