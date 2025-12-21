import streamlit as st
import pandas as pd
from nlp_processor import PolymerNLPProcessor
from logic import PolymerSelector

# Настройка страницы
st.set_page_config(page_title="Polymer Expert AI", layout="wide")

@st.cache_resource
def init_components():
    selector = PolymerSelector("data/polymers_with_names_predicted.csv")
    nlp = PolymerNLPProcessor(
        dataset_path="data/polymers_with_names_predicted.csv", 
        config_path="data/param_config.json"
    )
    return selector, nlp

selector, nlp = init_components()

# Словарь единиц измерения
UNITS_MAPPING = {
    "Egc": "кДж/моль", "Egb": "кДж/моль", "Eib": "кДж/моль", "CED": "Дж/см³",
    "Ei": "эВ", "Eea": "эВ", "Eat": "кДж/моль", "nc": "шт", "ne": "шт",
    "Xc": "%", "Xe": "моль/м³", "epsc": "0 Гц", "epsb": "кВ/мм",
    "TSb": "МПа", "TSy": "МПа", "YM": "МПа", "Cp": "Дж/(г·К)",
    "Td": "K", "Tg": "K", "Tm": "K", "rho": "г/см³", "LOI": "%"
}

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.title("Информационная панель")
    
    st.markdown("### Инструкция по поиску")
    st.info("""
    Система поддерживает запросы на естественном языке:
    
    1. Качественные характеристики: Используйте термины 'высокий', 'низкий', 'скромный'.
    2. Числовые параметры: Указывайте код и значение (например, Td 500).
    """)

    st.markdown("---")
    st.markdown("### Доступные параметры")
    
    search_query = st.text_input("Поиск по названию или коду", "").lower()
    
    # Формируем список параметров для отображения
    params_to_show = { 
        "Egc": "Group Contribution Energy", "Egb": "Bond Energy Contribution", "Eib": "Internal Bond Energy", 
        "CED": "Cohesive Energy Density", "Ei": "Ionization Energy", "Eea": "Electron Affinity", "nc": "Number of Carbon Atoms", "ne": "Number of Electrons", 
        "Xc": "Crystallinity Index", "Xe": "Crosslinking Density", "epse_6.0": "Dielectric Constant at 6.0 GHz", 
        "epsc": "Static Dielectric Constant", "epsb": "Breakdown Dielectric Strength", 
        "TSb": "Tensile Strength at Break", "TSy": "Tensile Strength at Yield", "YM": "Young's Modulus", 
        "Cp": "Specific Heat Capacity", "Td": "Thermal Decomposition Temperature", 
        "Tg": "температура стеклования", "Tm": "Melting Temperature", "rho": "Density", "LOI": "Limiting Oxygen Index"
    }

    for code, full_name_en in params_to_show.items():
        rus_name = nlp.config[code]['names'][0] if code in nlp.config else full_name_en
        unit = UNITS_MAPPING.get(code, "отн. ед.")
        
        if search_query in rus_name.lower() or search_query in code.lower():
            with st.container():
                st.markdown(f"**{code}**")
                st.markdown(f"*{rus_name}*")
                st.caption(f"Единица: {unit}")
                st.markdown("---")

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.title("Polymer Expert AI")
st.markdown("Система интеллектуального подбора полимерных материалов")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.markdown(msg["content"])
        else:
            st.dataframe(msg["content"], use_container_width=True, hide_index=True)

# Ввод пользователя
if prompt := st.chat_input("Введите описание необходимых свойств..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    extracted_reqs = nlp.extract_requirements(prompt)

    with st.chat_message("assistant"):
        if not extracted_reqs:
            st.warning("Требования не распознаны. Пожалуйста, используйте названия параметров из списка в боковой панели.")
        else:
            results = selector.find_best_matches(extracted_reqs, top_n=5)

            if results.empty:
                st.error("По вашему запросу совпадений не найдено.")
            else:
                explanation = "### Анализ запроса\n"
                for code, val in extracted_reqs.items():
                    name = nlp.config[code]['names'][0] if code in nlp.config else code
                    unit = UNITS_MAPPING.get(code, "")
                    explanation += f"- **{name.capitalize()}** ({code}): ~{val:.2f} {unit}\n"

                st.markdown(explanation)
                
                # Подготовка данных для вывода (скрываем лишнее)
                display_df = results.copy()

                st.markdown("### Результаты подбора")
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Name": st.column_config.TextColumn("Название", width="medium"),
                        "ID": st.column_config.TextColumn("Идентификатор", width="small"),
                        "similarity_score": None,  # Скрываем процент соответствия
                        "Polymer_SMILES": None,    # Скрываем структуру
                        "Monomer_SMILES_1": None,
                        "Monomer_SMILES_2": None
                    },
                    use_container_width=True,
                    hide_index=True
                )

                st.session_state.messages.append({"role": "assistant", "content": explanation})
                st.session_state.messages.append({"role": "assistant", "content": display_df})