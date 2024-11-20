rem RUN AS ADMINISTRATOR FROM WINDOWS
@echo off

rem Delete existing port proxy rule #########################################

netsh interface portproxy delete v4tov4 listenport=8555 listenaddress=0.0.0.0

rem Get WSL IP address ######################################################
FOR /F "tokens=*" %%i IN ('wsl -d Ubuntu hostname -I') DO SET WSL_IP=%%i

rem Trim spaces from WSL_IP #################################################
SET WSL_IP=%WSL_IP: =%
echo WSL IP: %WSL_IP%

rem Add port proxy rule #####################################################
netsh interface portproxy add v4tov4 listenport=8555 listenaddress=0.0.0.0 connectport=8555 connectaddress=%WSL_IP%

rem REM Start the Streamlit app directly in WSL
wsl -d Ubuntu -u kolommik bash -c "cd ~/GitHub/gh_speech_scriber/speech_scriber && ~/.local/bin/poetry run streamlit run app/main.py --server.headless true --server.port 8555"

rem no hang up (nohup) and suppress output & err
rem wsl -d Ubuntu -u kolommik bash -c "cd ~/GitHub/gh_speech_scriber/speech_scriber && ~/.local/bin/poetry run streamlit run app/main.py --server.headless true --server.port 8555 >/dev/null 2>&1 &"
