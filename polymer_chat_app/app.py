import streamlit as st
import re
from logic import PolymerSelector

st.set_page_config(page_title="Polymer Expert AI", layout="wide")

@st.cache_resource
def load_selector():
    # Укажи здесь путь к своему файлу
    return PolymerSelector("data/polymers_with_names_predicted.csv")

selector = load_selector()

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("Параметры поиска")
    
    st.subheader("Основные свойства")
    st.markdown("""
    - **Tg** — Температура стеклования
    - **Tm** — Температура плавления
    - **Td** — Температура разложения
    - **rho** — Плотность
    - **Egc** — Ширина запрещенной зоны (опт.)
    - **CED** — Плотность энергии когезии
    - **YM** — Модуль Юнга
    - **epsb** — Диэлектрическая проницаемость
    - **LOI** — Кислородный индекс
    - **Cp** — Теплоемкость
    """)

    with st.expander("Посмотреть все 40+ параметров"):
        # Достаем все колонки, кроме системных
        all_cols = selector.df.columns.tolist()
        system_cols = ['ID', 'Name', 'Polymer_SMILES', 'Monomer_SMILES_1', 'Monomer_SMILES_2', 'similarity_score']
        props_only = [c for c in all_cols if c not in system_cols]
        st.write(", ".join(props_only))

    st.divider()
    st.subheader("Примеры запросов")
    st.info("`Tg 250` \n\n `Tg 300 rho 1.2 Egc 4.0` \n\n `CED 400 YM 2000` ")

# --- ОСНОВНОЙ ИНТЕРФЕЙС ЧАТА ---
st.title("Поиск полимеров по заданным свойствам")
st.caption("Введите ваши требования к полимеру, и я подберу наиболее близкие варианты из базы.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.markdown(msg["content"])
        else:
            st.dataframe(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Напр: Tg 150 rho 1.05"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Логика извлечения параметров (Парсинг)
    # Ищем: Название_Свойства Число
    pairs = re.findall(r"([A-Za-z0-9_.]+)\s*[:=]*\s*([-+]?\d*\.\d+|\d+)", prompt)
    reqs = {k: float(v) for k, v in pairs}

    with st.chat_message("assistant"):
        if not reqs:
            error_msg = "Я не нашел параметров в вашем запросе. Пожалуйста, используйте формат: `Свойство Значение`. Список доступных свойств — слева в панели."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            results = selector.find_best_matches(reqs, top_n=5)
            
            if results.empty:
                error_msg = "К сожалению, в базе нет колонок с такими названиями. Посмотрите список доступных параметров в боковой панели."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                success_msg = f"Отобрал топ-5 полимеров, максимально близких к вашим значениям: `{reqs}`"
                st.success(success_msg)
                
                # Выводим таблицу
                st.dataframe(results, use_container_width=True)
                
                # Сохраняем в историю чата
                st.session_state.messages.append({"role": "assistant", "content": success_msg})
                st.session_state.messages.append({"role": "assistant", "content": results})