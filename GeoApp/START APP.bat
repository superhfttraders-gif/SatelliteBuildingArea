@echo off

cd /d D:\Projects\GeoApp

call .venv\Scripts\activate.bat

streamlit run app.py

pause