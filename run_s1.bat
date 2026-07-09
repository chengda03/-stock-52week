@echo off
REM ============================================================
REM  S1 불타기 전략 실운영 앱 실행기
REM  - 실행 시 오늘까지 시장데이터를 자동 증분 갱신한 뒤 앱을 띄웁니다.
REM ============================================================
cd /d %~dp0

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo [run_s1] Streamlit 앱을 실행합니다...
"%PY%" -m streamlit run app_s1.py

pause
