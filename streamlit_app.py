"""PeatTwin — Gemelo Digital de Turberas.

Aplicacion Streamlit que habla directamente con PostgreSQL: no hay API HTTP ni
contenedores. Arranque desde la raiz del repositorio:

    streamlit run streamlit_app.py

Requiere la base de datos migrada (`alembic upgrade head`) y, para las vistas de
series, modelos y simulador, los datos y el modelo que produce el modulo `ml/`.
"""
from __future__ import annotations

import streamlit as st

from ui import auth, theme
from views import admin, login, modelos, motor_ia, series, simulador

st.set_page_config(
    page_title="PeatTwin",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Capa visual. Va antes de cualquier render, incluida la pantalla de acceso, y
# solo aporta CSS: sin ella la aplicacion funciona igual con el tema por
# defecto de Streamlit.
theme.inject()


def _sidebar() -> None:
    user = auth.current_user()
    with st.sidebar:
        st.markdown("### 🌱 PeatTwin")
        st.caption("Gemelo Digital de Turberas")
        st.divider()
        st.markdown(f"**{user.display_name}**")
        st.caption(f"@{user.username} · {' · '.join(user.role_names) or 'sin rol'}")
        if st.button("Cerrar sesion", width="stretch"):
            auth.logout()
            st.rerun()


def main() -> None:
    if not auth.is_authenticated():
        # Tambien aqui va por `st.navigation`, con el menu oculto: si no, al
        # cerrar sesion el frontend seguiria mostrando la navegacion anterior.
        st.navigation(
            [st.Page(login.render, title="Entrar", url_path="entrar")],
            position="hidden",
        ).run()
        return

    _sidebar()

    # `url_path` explicito: las cuatro vistas exponen una funcion `render`, y sin
    # el Streamlit derivaria la misma ruta de `__name__` para todas.
    pages = [
        st.Page(series.render, title="Series temporales", url_path="series",
                icon=":material/show_chart:", default=True),
        st.Page(modelos.render, title="Comparacion de modelos", url_path="modelos",
                icon=":material/table_chart:"),
        st.Page(simulador.render, title="Simulador de escenarios", url_path="simulador",
                icon=":material/tune:"),
    ]
    if auth.has_role("admin"):
        pages.append(
            st.Page(motor_ia.render, title="Motor IA", url_path="motor-ia",
                    icon=":material/neurology:")
        )
        pages.append(
            st.Page(admin.render, title="Usuarios y roles", url_path="admin",
                    icon=":material/group:")
        )

    st.navigation(pages).run()


main()
