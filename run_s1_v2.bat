@echo off
chcp 65001 >nul
cd /d "C:\Users\leesungdae\Desktop\-stock-52week"
call .venv\Scripts\activate.bat
streamlit run app_s1.py
pause
