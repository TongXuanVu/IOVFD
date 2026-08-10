@echo off
REM Commit va day repo IOVFD len GitHub.
REM   push.bat "mo ta thay doi"
setlocal
set MSG=%~1
if "%MSG%"=="" set MSG=Cap nhat
if not exist .git (
  git init -b main
  git remote add origin https://github.com/TongXuanVu/IOVFD.git
)
git add -A
git commit -m "%MSG%" || echo Khong co gi moi de commit
git push -u origin main
endlocal
