"""Camada de apresentação do portal clínico.

Separada de `app.py` para que o entry point cuide só de layout e interação,
enquanto formatação de saída, rótulos de negócio e tema ficam em módulos
testáveis sem subir o Streamlit. Nada aqui importa `app.py`, o que mantém
estes módulos reutilizáveis por qualquer outra interface.
"""
